# -*- coding: UTF-8 -*-

"""
@author: zhangjie
@software: Pycharm
@license: Apache Licence
@file: train.py
@time: 2022/10/8 2:47 下午
@desc:
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from config import Config
cfg = Config()

import datetime
import numpy as np
import torch
from scipy.stats.stats import pearsonr
from utils import obtain_samples, moving_mean
from model import SequentialModel
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    model = SequentialModel(cfg).to(device)

    best_test_loss = 0.0
    best_test_pcc = 0.0
    best_epoch = 0
    mse_list = []
    pcc_list = []
    for epoch in range(1):
        """
        torch.cuda.empty_cache()
        train_step = 0
        movies = 0
        train_data_list = np.load(cfg.train_path, allow_pickle=True)
        for train_data in train_data_list:
            movies += 1
            vggish_features, bgm_emotion_features, emotion_features, \
            scene_features, pose_features, labels_ori = train_data
            model.train()
            model.zero_grad()
            # optimizer.zero_grad()
            assert vggish_features.shape[0] == emotion_features.shape[0] == \
                   scene_features.shape[0] == pose_features.shape[0]

            # train_step += 1
            # if cfg.task == 'arousal':
            #     labels_ori = arousalValue
            # else:
            #     labels_ori = valenceValue

            samples = obtain_samples(vggish_features, bgm_emotion_features, emotion_features, scene_features,
                                     pose_features, labels_ori, cfg.time_steps, device)

            h = None
            for step_ in range(len(samples)):
                train_step += 1
                vggish_data, bgm_emotion_data, emotion_data, scene_data, pose_data, labels = samples[step_]
                # pose_data = pose_data.to(torch.float32)
                if step_ == len(samples) - 1:
                    next_vggish, next_bgm_emotion, next_emotion, next_scene, next_pose, next_labels = samples[step_]
                    end_flag = torch.zeros_like(next_labels)
                else:
                    next_vggish, next_bgm_emotion, next_emotion, next_scene, next_pose, next_labels = samples[step_ + 1]
                    end_flag = torch.ones_like(next_labels)

                h, y = model.forward(vggish_data, bgm_emotion_data, emotion_data, scene_data, pose_data,
                                        next_vggish, next_bgm_emotion, next_emotion, next_scene, next_pose,
                                        end_flag, h, labels, next_labels, step_, epoch, train_step, len(samples), movies,
                                        training=True)
        """
        print('start test')
        test_step = 0
        test_mse = 0.0
        test_pcc = 0.0
        model.load_state_dict(torch.load('./pre_training/best_model_smooth_1664_pcc0.427_mse0.00915.pt'))
        model.eval()
        with torch.no_grad():
            test_data_list = np.load(cfg.test_path, allow_pickle=True)

            gt_labels_list = []
            pre_labels_list = []
            for test_data in test_data_list:
                vggish_features, bgm_emotion_features, emotion_features, \
                scene_features, pose_features, labels_ori = test_data
                assert vggish_features.shape[0] == emotion_features.shape[0] == \
                       scene_features.shape[0] == pose_features.shape[0]
                test_step += 1
                # if cfg.task == 'arousal':
                #     labels_ori = arousalValue
                # else:
                #     labels_ori = valenceValue
                samples = obtain_samples(vggish_features, bgm_emotion_features, emotion_features, scene_features,
                                         pose_features, labels_ori, cfg.time_steps, device)

                h = None
                pre_labels = []
                for step_ in range(len(samples)):
                    vggish_data, bgm_emotion_data, emotion_data, scene_data, pose_data, labels = samples[step_]
                    h, y = model.forward(vggish_data, bgm_emotion_data, emotion_data, scene_data,
                                            pose_data, None, None, None, None, None, None, h, None, None,
                                            step_, epoch, test_step, len(samples), 0, training=False)
                    pre_labels.append(y)

                pre_labels = torch.cat(pre_labels, dim=0).cpu().data.numpy()
                pre_labels = np.tile(pre_labels, (1, cfg.time_steps))
                pre_labels = np.reshape(pre_labels, (-1,))
                gt_labels = np.reshape(labels_ori, (-1, ))[:pre_labels.shape[0]]
                if cfg.moving_mean:
                    pre_labels = moving_mean(pre_labels, cfg.moving_mean_value)
                movie_pcc = pearsonr(pre_labels, gt_labels)[0]
                test_pcc += movie_pcc
                movie_mse = np.mean(np.square(gt_labels - pre_labels))
                test_mse += movie_mse
                print(f'test_step:{test_step}, pcc:{movie_pcc}, mse:{movie_mse}')

                gt_labels_list.append(gt_labels)
                pre_labels_list.append(pre_labels)
                pcc_list.append(movie_pcc)
                mse_list.append(movie_mse)

            dic = {}
            dic['gt_labels_list'] = gt_labels_list
            dic['pre_labels_list'] = pre_labels_list
            dic['pcc_list'] = pcc_list
            dic['mse_list'] = mse_list
            with open('./result/gt_pre_smooth_1664.pkl', 'wb') as fo:
                pickle.dump(dic, fo, protocol=4)
                fo.close()

            test_pcc = test_pcc / test_step
            test_mse = test_mse / test_step

            if test_pcc > best_test_pcc:
                best_test_loss = test_mse
                best_test_pcc = test_pcc
                best_epoch = epoch

            now_time = datetime.datetime.now()
            str_time = now_time.strftime("%Y-%m-%d %H:%M:%S")
            test_log_str = '\r[%s] Epoch:%d, movies:%d test_loss:%.3f, test_pcc:%.3f' \
                           ', best_test_loss:%.3f, best_test_pcc:%.3f, best_epoch:%d' \
                           % (str_time, epoch, test_step, test_mse, test_pcc,
                              best_test_loss, best_test_pcc, best_epoch)
            print(test_log_str)
            exit()
            # torch.save(model.state_dict(), f'./pre_training/epoch{epoch}_pcc{test_pcc}_mse{test_mse}.pt')
            # print(f'Save ./pre_training/epoch{epoch}_pcc{test_pcc}_mse{test_mse}.pt')



if __name__ == "__main__":
    train()
