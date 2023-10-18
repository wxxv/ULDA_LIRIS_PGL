# -*- coding: UTF-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch.nn as nn
import torch
import torch.nn.functional as F
import random
import numpy as np
random.seed(1000)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAEarlyFusion(nn.Module):
    def __init__(self, cfg=None):
        super(VAEarlyFusion, self).__init__()
        self.cfg = cfg

        # 特征降维
        self.vggish_dense = nn.Linear(self.cfg.vggish_input_dim, self.cfg.modals_ouput_dim)
        self.bgm_emotion_dense = nn.Linear(self.cfg.bgm_emotion_input_dim, self.cfg.modals_ouput_dim)
        self.emotion_dense = nn.Linear(self.cfg.emotion_input_dim, self.cfg.modals_ouput_dim)
        self.scene_dense = nn.Linear(self.cfg.scene_input_dim, self.cfg.modals_ouput_dim)
        self.pose_dense = nn.Linear(self.cfg.pose_input_dim, self.cfg.modals_ouput_dim)

        # 模态融合
        self.modal_fusion_dense = nn.Linear(self.cfg.modals_ouput_dim * (len(self.cfg.modals) - 1), self.cfg.modals_ouput_dim)
        # 时序融合
        self.temporal_lstm = nn.LSTM(self.cfg.modals_ouput_dim,
                                     self.cfg.modals_ouput_dim,
                                     num_layers=self.cfg.lstm_layers,
                                     bidirectional=self.cfg.bidirectional)
        self.temporal_dense = nn.Linear(2 * self.cfg.modals_ouput_dim, self.cfg.modals_ouput_dim)

        # 预测层
        self.predict_dense = nn.Linear(self.cfg.modals_ouput_dim, 1)

    def forward(self, vggish, bgm_emotion, emotion, scene, pose):
        modals_list = []
        if 'vggish' in self.cfg.modals:
            vggish = torch.reshape(vggish, (-1, self.cfg.time_steps, self.cfg.vggish_input_dim))
            vggish = self.vggish_dense(vggish).relu()
            modals_list.append(vggish)
        # if 'bgm_emotion' in self.cfg.modals:
        #     bgm_emotion = torch.reshape(bgm_emotion, (-1, self.cfg.time_steps, self.cfg.bgm_emotion_input_dim))
        #     bgm_emotion = self.bgm_emotion_dense(bgm_emotion).relu()
        #     modals_list.append(bgm_emotion)
        if 'emotion' in self.cfg.modals:
            emotion = torch.reshape(emotion, (-1, self.cfg.time_steps, self.cfg.emotion_input_dim))
            emotion = self.emotion_dense(emotion).relu()
            modals_list.append(emotion)
        if 'scene' in self.cfg.modals:
            scene = torch.reshape(scene, (-1, self.cfg.time_steps, self.cfg.scene_input_dim))
            scene = self.scene_dense(scene).relu()
            modals_list.append(scene)
        if 'pose' in self.cfg.modals:
            pose = torch.reshape(pose, (-1, self.cfg.time_steps, self.cfg.pose_input_dim))
            pose = self.pose_dense(pose).relu()
            modals_list.append(pose)

        fused_modals = torch.cat(modals_list, dim=-1)
        fused_modals = self.modal_fusion_dense(fused_modals).relu()

        x, (_, _) = self.temporal_lstm(fused_modals.permute(1, 0, 2))
        x = x[-1, ...]
        # _, (h, _) = self.temporal_lstm(fused_modals.permute(1, 0, 2))
        # x = h[-2:].permute(1, 0, 2).reshape((h.size(1), -1))
        x = self.temporal_dense(x).relu()
        y = self.predict_dense(x).tanh()
        return y, x


class FixedMemoryModule(nn.Module):
    def __init__(self, cfg=None):
        super(FixedMemoryModule, self).__init__()
        self.cfg = cfg
        self.weight_dense_read = nn.Sequential(
            nn.Linear(self.cfg.feature_dim * 2, self.cfg.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.cfg.feature_dim, 1)
        )

        self.weight_dense_write = nn.Sequential(
            nn.Linear(self.cfg.feature_dim * 2, self.cfg.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.cfg.feature_dim, 1)
        )

        self.sub_f = nn.Linear(self.cfg.feature_dim * 2, self.cfg.feature_dim)
        self.sub_i = nn.Linear(self.cfg.feature_dim * 2, self.cfg.feature_dim)
        # self.sub_o = nn.Linear(self.cfg.feature_dim * 2, self.cfg.feature_dim)
        self.sub_c_in = nn.Linear(self.cfg.feature_dim * 2, self.cfg.feature_dim)

        self.sub_dense = nn.Linear(self.cfg.feature_dim, self.cfg.feature_dim)
        self.final_dense = nn.Linear(self.cfg.feature_dim, self.cfg.feature_dim)

    def content_based_addressing_read(self, x, story_memory):
        story_weight = self.weight_dense_read(torch.cat([story_memory, x], dim=-1))
        story_weight = torch.softmax(story_weight, dim=1)
        return story_weight

    def content_based_addressing_write(self, x, story_memory):
        story_weight = self.weight_dense_write(torch.cat([story_memory, x], dim=-1))
        story_weight = torch.softmax(story_weight, dim=1)
        return story_weight

    def content_based_addressing(self, x, story_memory):
        norm_x = x / torch.norm(x, dim=-1, keepdim=True)
        norm_memory = story_memory / torch.norm(story_memory, dim=-1, keepdim=True)
        weight = torch.sum(norm_x * norm_memory, dim=-1, keepdim=True)
        weight = torch.softmax(weight * 10, dim=1)
        return weight

    def read(self, x_ori, sub_memory):
        x = torch.unsqueeze(x_ori, dim=1).repeat((1, sub_memory.size(1), 1))
        sim = self.content_based_addressing_read(x, sub_memory)
        # sub_o = self.sub_o(torch.cat([sub_memory, x], dim=-1)).sigmoid()
        sub_vector = torch.sum(sub_memory * sim, dim=1)
        final_vector = self.final_dense(sub_vector).tanh()
        return final_vector

    def write(self, x_ori, sub_memory, stage='train'):

        x = torch.unsqueeze(x_ori, dim=1).repeat((1, sub_memory.size(1), 1))
        sim = self.content_based_addressing_write(x, sub_memory)
        if stage == 'behavior' and self.cfg.noise_method == 'random':
            random_noise = torch.rand((1, ))
            if random_noise < self.cfg.noise_rate:
                sim = torch.rand_like(sim)
                sim = torch.softmax(torch.pow(sim + 1.0, 2), dim=1)
        sub_f = self.sub_f(torch.cat([sub_memory, x], dim=-1)).sigmoid()
        sub_i = self.sub_i(torch.cat([sub_memory, x], dim=-1)).sigmoid()
        if stage == 'behavior' and self.cfg.noise_method == 'noise':
            random_noise_1 = torch.rand((1,))
            if random_noise_1 < self.cfg.noise_rate:
                sim = sim + torch.randn_like(sim) * self.cfg.gaussian_std + self.cfg.gaussian_mean
                sim = torch.relu(sim)
                sim = sim / torch.sum(sim, dim=1, keepdim=True)
            random_noise_2 = torch.rand((1,))
            if random_noise_2 < self.cfg.noise_rate:
                sub_f = sub_f + torch.randn_like(sub_f) * self.cfg.gaussian_std + self.cfg.gaussian_mean
                sub_f = torch.clamp(sub_f, min=0.0, max=1.0)
            random_noise_3 = torch.rand((1,))
            if random_noise_3 < self.cfg.noise_rate:
                sub_i = sub_i + torch.randn_like(sub_i) * self.cfg.gaussian_std + self.cfg.gaussian_mean
                sub_i = torch.clamp(sub_i, min=0.0, max=1.0)

        sub_c_in = self.sub_c_in(torch.cat([sub_memory, x], dim=-1)).tanh()
        sub_memory = (-sim * sub_f + 1.0) * sub_memory + sub_i * sub_c_in * sim
        sub_memory = self.sub_dense(sub_memory).tanh()
        return sub_memory


class Actor(nn.Module):
    def __init__(self, cfg=None):
        super(Actor, self).__init__()
        self.cfg = cfg
        self.base_model = VAEarlyFusion(self.cfg)
        if self.cfg.base_model_weights_path is not None:
            print('load base model weights')
            self.base_model.load_state_dict(torch.load(self.cfg.base_model_weights_path), strict=False)
            print('finish load base model weights')

        self.memory_module = FixedMemoryModule(self.cfg)

        self.pre_dense = nn.Linear(self.cfg.modals_ouput_dim, 1)

    def forward(self, vggish, bgm_emotion, emotion, scene, pose, sub_memory, target, epochs, training, stage='train'):
        _, x = self.base_model.forward(vggish, bgm_emotion, emotion, scene, pose)
        sub_memory = self.memory_module.write(x, sub_memory, stage=stage)
        x = self.memory_module.read(x, sub_memory)
        y = self.pre_dense(x).sigmoid()
        return sub_memory, y


class Value(nn.Module):
    def __init__(self, cfg=None):
        super(Value, self).__init__()
        self.cfg = cfg

        self.base_model = VAEarlyFusion(self.cfg)
        if self.cfg.base_model_weights_path is not None:
            print('load base model weights')
            self.base_model.load_state_dict(torch.load(self.cfg.base_model_weights_path), strict=False)
            print('finish load base model weights')

        self.memory_module = FixedMemoryModule(self.cfg)

        self.pre_dense = nn.Linear(self.cfg.modals_ouput_dim, 1)

    def forward(self, vggish, bgm_emotion, emotion, scene, pose, sub_memory, target, epochs, training):
        _, x = self.base_model.forward(vggish, bgm_emotion, emotion, scene, pose)
        final_vector = self.memory_module.read(x, sub_memory)
        y = self.pre_dense(final_vector).tanh()
        return y


class SequentialModel(nn.Module):
    def __init__(self, cfg=None):
        super(SequentialModel, self).__init__()
        self.cfg = cfg

        self.model = Actor(cfg)
        self.target_model = Actor(cfg)
        for p in self.target_model.parameters():
            p.requires_grad = False
        self.target_model.load_state_dict(self.model.state_dict())

        self.q = Value(cfg)
        self.target_q = Value(cfg)
        for p in self.target_q.parameters():
            p.requires_grad = False
        self.target_q.load_state_dict(self.q.state_dict())

        self.sub_memory_initial = nn.Parameter(torch.randn((1, self.cfg.memory_num, self.cfg.feature_dim)).to(device),
                                               requires_grad=True)

        self.memory_bank = nn.Parameter(torch.randn((1, self.cfg.memory_num, self.cfg.feature_dim)).to(device),
                                        requires_grad=True)

        self.buf = []
        self.buf_num = self.cfg.buf_num

        self.train_steps = self.cfg.train_steps
        self.batch_size = self.cfg.batch_size

        # self.opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=self.cfg.q_lr)
        self.pi_opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.pi_lr)

        self.tao = self.cfg.tao

    def soft_update_params(self):
        target_model_keys = self.target_model.state_dict().keys()

        new_target_model_map = {}
        model_map = self.model.state_dict()
        target_model_map = self.target_model.state_dict()
        for target_model_key in target_model_keys:
            new_target_model_map[target_model_key] = model_map[target_model_key] * self.tao + (1 - self.tao) * target_model_map[target_model_key]
        self.target_model.load_state_dict(new_target_model_map)

        new_target_q_map = {}
        q_keys = self.target_q.state_dict().keys()
        q_map = self.q.state_dict()
        target_q_map = self.target_q.state_dict()
        for q_key in q_keys:
            new_target_q_map[q_key] = q_map[q_key] * self.tao + (1 - self.tao) * target_q_map[q_key]

        self.target_q.load_state_dict(new_target_q_map)

    def sample_from_buf(self):
        self.buf = self.buf[-self.buf_num:]
        random.shuffle(self.buf)
        x_list = self.buf[:self.batch_size]

        pre_h_batch = []

        vggish_batch = []
        bgm_emotion_batch = []
        emotion_batch = []
        scene_batch = []
        pose_batch = []

        next_vggish_batch = []
        next_bgm_emotion_batch = []
        next_emotion_batch = []
        next_scene_batch = []
        next_pose_batch = []
        end_flag_batch = []

        h_batch = []
        y_batch = []
        target_batch = []
        next_target_batch = []

        for i in range(len(x_list)):
            pre_h, vggish, bgm_emotion, emotion, scene, pose, next_vggish, next_bgm_emotion, next_emotion,\
            next_scene, next_pose, end_flag, h, y, target, next_target = x_list[i]
            pre_h_batch.append(pre_h)

            vggish_batch.append(vggish)
            bgm_emotion_batch.append(bgm_emotion)
            emotion_batch.append(emotion)
            scene_batch.append(scene)
            pose_batch.append(pose)

            next_vggish_batch.append(next_vggish)
            next_bgm_emotion_batch.append(next_bgm_emotion)
            next_emotion_batch.append(next_emotion)
            next_scene_batch.append(next_scene)
            next_pose_batch.append(next_pose)
            end_flag_batch.append(end_flag)

            h_batch.append(h)
            y_batch.append(y)
            target_batch.append(target)
            next_target_batch.append(next_target)
        pre_h_batch = torch.cat(pre_h_batch, dim=0)

        vggish_batch = torch.cat(vggish_batch, dim=0)
        bgm_emotion_batch = torch.cat(bgm_emotion_batch, dim=0)
        emotion_batch = torch.cat(emotion_batch, dim=0)
        scene_batch = torch.cat(scene_batch, dim=0)
        pose_batch = torch.cat(pose_batch, dim=0)

        next_vggish_batch = torch.cat(next_vggish_batch, dim=0)
        next_bgm_emotion_batch = torch.cat(next_bgm_emotion_batch, dim=0)
        next_emotion_batch = torch.cat(next_emotion_batch, dim=0)
        next_scene_batch = torch.cat(next_scene_batch, dim=0)
        next_pose_batch = torch.cat(next_pose_batch, dim=0)
        end_flag_batch = torch.cat(end_flag_batch, dim=0)

        y_batch = torch.cat(y_batch, dim=0)
        h_batch = torch.cat(h_batch, dim=0)

        target_batch = torch.cat(target_batch, dim=0)
        next_target_batch = torch.cat(next_target_batch, dim=0)
        return pre_h_batch, vggish_batch, bgm_emotion_batch, emotion_batch, scene_batch, pose_batch,\
               next_vggish_batch, next_bgm_emotion_batch, next_emotion_batch, next_scene_batch,\
               next_pose_batch, end_flag_batch, h_batch, y_batch, target_batch, next_target_batch

    def forward(self, vggish, bgm_emotion, emotion, scene, pose, next_vggish, next_bgm_emotion,
                next_emotion, next_scene, next_pose, end_flag, pre_h, target, next_target, steps, epochs, train_step, l, movies,
                training=True):
        self.pi_opt.zero_grad()
        self.q_opt.zero_grad()
        if not training:
            if steps == 0:
                h, y = self.model(vggish, bgm_emotion, emotion, scene, pose,
                                  self.sub_memory_initial, target, epochs, training)
            else:
                h, y = self.model(vggish, bgm_emotion, emotion, scene, pose, pre_h, target, epochs, training)

        else:
            buf = []
            if steps == 0:
                h, y = self.target_model(vggish, bgm_emotion, emotion, scene, pose,
                                         self.sub_memory_initial, target, epochs, training, stage='behavior')
                buf.extend([self.sub_memory_initial, vggish,
                            bgm_emotion, emotion, scene, pose, next_vggish,
                            next_bgm_emotion, next_emotion, next_scene, next_pose, end_flag,
                            h.detach_(), y.detach_(), target, next_target])
            else:
                h, y = self.target_model(vggish, bgm_emotion, emotion, scene, pose, pre_h, target,
                                         epochs, training, stage='behavior')
                buf.extend([pre_h, vggish, bgm_emotion, emotion, scene, pose, next_vggish,
                            next_bgm_emotion, next_emotion, next_scene, next_pose, end_flag,
                            h.detach_(), y.detach_(), target, next_target])
            self.buf.append(buf)

            if train_step % self.train_steps == 0:

                # 采样
                pre_h_batch, vggish_batch, bgm_emotion_batch, emotion_batch, scene_batch, pose_batch, \
                next_vggish_batch, next_bgm_emotion_batch, next_emotion_batch, next_scene_batch, next_pose_batch,\
                end_flag_batch, h_batch, y_batch, target_batch, next_target_batch = self.sample_from_buf()

                # 更新Q
                r = - torch.abs(y_batch - target_batch) + 1.0

                next_h, next_y = self.target_model(next_vggish_batch, next_bgm_emotion_batch, next_emotion_batch,
                                                        next_scene_batch, next_pose_batch,
                                                        h_batch, next_target_batch, epochs, training)
                max_q = self.target_q(next_vggish_batch, next_bgm_emotion_batch, next_emotion_batch, next_scene_batch,
                                      next_pose_batch, next_h, next_target_batch, epochs, training)
                target_value = r + self.cfg.q_factor * max_q * end_flag_batch
                value = self.q(vggish_batch, bgm_emotion_batch, emotion_batch, scene_batch, pose_batch,
                               h_batch, target_batch, epochs, training)
                q_loss = F.mse_loss(value, target_value.float())
                q_loss.backward()
                self.q_opt.step()
                self.q_opt.zero_grad()

                # 更新pi
                new_h, new_y = self.model(vggish_batch, bgm_emotion_batch, emotion_batch, scene_batch, pose_batch,
                                                 pre_h_batch, target_batch, epochs, training)

                new_next_h, new_next_y = self.target_model(next_vggish_batch, next_bgm_emotion_batch,
                                                                next_emotion_batch, next_scene_batch,
                                                                next_pose_batch, new_h,
                                                                next_target_batch, epochs, training)

                new_r = - torch.abs(new_y - target_batch) + 1.0
                new_max_q = self.q(next_vggish_batch, next_bgm_emotion_batch, next_emotion_batch, next_scene_batch,
                                   next_pose_batch, new_next_h, next_target_batch, epochs, training)
                loss = - torch.mean((new_r + self.cfg.pi_factor * new_max_q * end_flag_batch))

                # print('train steps:%.2f / %d, q_loss:%.3f, pi_loss:%.3f' % (steps, l, q_loss.cpu().data.numpy(), loss.cpu().data.numpy()))
                if movies % 5 == 0:
                    print(f'epoch : {epochs}, movies : {movies}, train steps : {steps:.2f} / {l}, q_loss : {q_loss.cpu().data.numpy():.3f}, pi_loss : {loss.cpu().data.numpy():.3f}')

                loss.backward()
                self.pi_opt.step()
                self.pi_opt.zero_grad()

                # 更新target网络
                self.soft_update_params()
                with open(f'result/txt/q_smooth_gau.txt', 'a') as f:
                    f.write(str(np.mean(new_max_q.reshape(-1).detach().cpu().numpy(), axis=-1)) + '\n')
                with open(f'./result/txt/piloss_smooth_gau.txt', 'a') as f:
                    f.write(str(loss) + '\n')

            # if train_step % self.train_steps == 0:
            #     if movies % 5 == 0:
            #         print(f'epoch : {epochs}, movies : {movies}, train steps : {steps:.2f} / {l}, q_loss : {q_loss.cpu().data.numpy():.3f}, pi_loss : {loss.cpu().data.numpy():.3f}')

        return h.detach_(), y.detach_()

