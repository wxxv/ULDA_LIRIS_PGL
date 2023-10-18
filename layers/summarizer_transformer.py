# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.attention import SelfAttention
from data_augment_weight_gau import data_augment
import sys
from config import Config
import pickle
from sklearn.decomposition import PCA

sys.path.append("..")

class MultiAttention(nn.Module):
    def __init__(self, input_size=1024, output_size=1024, freq=10000, pos_enc=None,
                 num_segments=None, heads=1, fusion=None):
        """ Class wrapping the MultiAttention part of PGL-SUM; its key modules and parameters.

        :param int input_size: The expected input feature size.
        :param int output_size: The hidden feature size of the attention mechanisms.
        :param int freq: The frequency of the sinusoidal positional encoding.
        :param None | str pos_enc: The selected positional encoding [absolute, relative].
        :param None | int num_segments: The selected number of segments to split the videos.
        :param int heads: The selected number of global heads.
        :param None | str fusion: The selected type of feature fusion.
        """
        super(MultiAttention, self).__init__()

        # Global Attention, considering differences among all frames
        self.encoder_layer_global = nn.TransformerEncoderLayer(d_model=input_size, nhead=4)
        self.transformer_encoder_global = nn.TransformerEncoder(self.encoder_layer_global, num_layers=2)


        self.num_segments = num_segments
        if self.num_segments is not None:
            assert self.num_segments >= 2, "num_segments must be None or 2+"
            self.local_attention = nn.ModuleList()
            for _ in range(self.num_segments):
                # Local Attention, considering differences among the same segment with reduce hidden size
                self.local_attention.append(nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=input_size, nhead=4), num_layers=2))
        self.permitted_fusions = ["add", "mult", "avg", "max"]
        self.fusion = fusion
        if self.fusion is not None:
            self.fusion = self.fusion.lower()
            assert self.fusion in self.permitted_fusions, f"Fusion method must be: {*self.permitted_fusions,}"

    def forward(self, x):
        """ Compute the weighted frame features, based on the global and locals (multi-head) attention mechanisms.

        :param torch.Tensor x: Tensor with shape [T, input_size] containing the frame features.
        :return: A tuple of:
            weighted_value: Tensor with shape [T, input_size] containing the weighted frame features.
            attn_weights: Tensor with shape [T, T] containing the attention weights.
        """
        weighted_value = self.transformer_encoder_global(x)  # global attention
        if self.num_segments is not None and self.fusion is not None:
            segment_size = math.ceil(x.shape[0] / self.num_segments)
            for segment in range(self.num_segments):
                left_pos = segment * segment_size
                right_pos = (segment + 1) * segment_size
                local_x = x[left_pos:right_pos]
                weighted_local_value = self.local_attention[segment](local_x)  # local attentions

                # Normalize the features vectors
                weighted_value[left_pos:right_pos] = F.normalize(weighted_value[left_pos:right_pos].clone(), p=2, dim=1)
                # weighted_local_value = F.normalize(weighted_local_value, p=2, dim=1)
                if self.fusion == "add":
                    weighted_value[left_pos:right_pos] += weighted_local_value
                elif self.fusion == "mult":
                    weighted_value[left_pos:right_pos] *= weighted_local_value
                elif self.fusion == "avg":
                    weighted_value[left_pos:right_pos] += weighted_local_value
                    weighted_value[left_pos:right_pos] /= 2
                elif self.fusion == "max":
                    weighted_value[left_pos:right_pos] = torch.max(weighted_value[left_pos:right_pos].clone(),
                                                                   weighted_local_value)

        return weighted_value, 0


class PGL_SUM(nn.Module):
    def __init__(self, input_size=256, output_size=128, freq=10000, pos_enc=None,
                 num_segments=None, heads=8, fusion=None):
        """ Class wrapping the PGL-SUM model; its key modules and parameters.

        :param int input_size: The expected input feature size.
        :param int output_size: The hidden feature size of the attention mechanisms.
        :param int freq: The frequency of the sinusoidal positional encoding.
        :param None | str pos_enc: The selected positional encoding [absolute, relative].
        :param None | int num_segments: The selected number of segments to split the videos.
        :param int heads: The selected number of global heads.
        :param None | str fusion: The selected type of feature fusion.
        """
        super(PGL_SUM, self).__init__()
        self.cfg = Config()
        self.dict_new = pickle.load(open('./dict_new/dict_new_list32.pkl', 'rb'))
        # 特征降维
        self.vggish_dense = nn.Linear(self.cfg.vggish_input_dim, self.cfg.modals_ouput_dim)
        self.bgm_emotion_dense = nn.Linear(self.cfg.bgm_emotion_input_dim, self.cfg.modals_ouput_dim)
        self.emotion_dense = nn.Linear(self.cfg.emotion_input_dim, self.cfg.modals_ouput_dim)
        self.scene_dense = nn.Linear(self.cfg.scene_input_dim, self.cfg.modals_ouput_dim)
        self.pose_dense = nn.Linear(self.cfg.pose_input_dim, self.cfg.modals_ouput_dim)
        # 模态融合
        self.modal_fusion_dense = nn.Linear(self.cfg.modals_ouput_dim * len(self.cfg.modals),
                                            self.cfg.modals_ouput_dim)
        self.temporal_lstm = nn.LSTM(self.cfg.modals_ouput_dim,
                                     self.cfg.modals_ouput_dim,
                                     num_layers=self.cfg.lstm_layers,
                                     bidirectional=self.cfg.bidirectional)

        self.attention = MultiAttention(input_size=input_size, output_size=output_size, freq=freq,
                                        pos_enc=pos_enc, num_segments=num_segments, heads=heads, fusion=fusion)
        self.linear_0 = nn.Linear(in_features=input_size, out_features=input_size//4)
        self.linear_1 = nn.Linear(in_features=input_size//4, out_features=input_size//4)
        self.linear_2 = nn.Linear(in_features=self.linear_1.out_features, out_features=1)

        self.drop = nn.Dropout(p=0.5)
        self.norm_y = nn.LayerNorm(normalized_shape=input_size, eps=1e-6)
        self.norm_linear = nn.LayerNorm(normalized_shape=self.linear_1.out_features, eps=1e-6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, frame_features, target, aug, epoch, movies):
        """ Produce frames importance scores from the frame features, using the PGL-SUM model.

        :param torch.Tensor frame_features: Tensor of shape [T, input_size] containing the frame features produced by
        using the pool5 layer of GoogleNet.
        :return: A tuple of:
            y: Tensor with shape [1, T] containing the frames importance scores in [0, 1].
            attn_weights: Tensor with shape [T, T] containing the attention weights.
        """
        vggish, bgm_emotion, emotion, scene, pose = frame_features
        modals_list = []
        if 'vggish' in self.cfg.modals:
            vggish = vggish.unsqueeze(0)
            vggish = self.vggish_dense(vggish).relu()
            modals_list.append(vggish)
        if 'bgm_emotion' in self.cfg.modals:
            bgm_emotion = bgm_emotion.unsqueeze(0)
            bgm_emotion = self.bgm_emotion_dense(bgm_emotion).relu()
            modals_list.append(bgm_emotion)
        if 'emotion' in self.cfg.modals:
            emotion = emotion.unsqueeze(0)
            emotion = self.emotion_dense(emotion).relu()
            modals_list.append(emotion)
        if 'scene' in self.cfg.modals:
            scene = scene.unsqueeze(0)
            scene = self.scene_dense(scene).relu()
            modals_list.append(scene)
        if 'pose' in self.cfg.modals:
            pose = pose.unsqueeze(0)
            pose = self.pose_dense(pose).relu()
            modals_list.append(pose)

        fused_modals = torch.cat(modals_list, dim=-1)
        fused_modals = self.modal_fusion_dense(fused_modals).relu()

        x, (_, _) = self.temporal_lstm(fused_modals.permute(1, 0, 2))
        frame_features = x.squeeze(1)

        residual = frame_features
        weighted_value, attn_weights = self.attention(frame_features)
        y = weighted_value + residual
        y = self.drop(y)
        y = self.norm_y(y)

        y = self.linear_0(y)
        y = self.drop(y)
        # pca = PCA(n_components=64)
        # y = pca.fit_transform(y.detach().cpu())
        # y = torch.Tensor(y).cuda(1)

        # where the algorithm works
        if self.training and aug and epoch%1==0:
            y, aug_target = data_augment(y, target, movies, dict_new=self.dict_new)
        else:
            aug_target = target

        # 2-layer NN (Regressor Network)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.drop(y)
        y = self.norm_linear(y)

        y = self.linear_2(y)
        y = self.sigmoid(y)
        y = y.view(1, -1)

        return y, attn_weights, aug_target


if __name__ == '__main__':
    pass
    """Uncomment for a quick proof of concept
    model = PGL_SUM(input_size=256, output_size=256, num_segments=3, fusion="Add").cuda()
    _input = torch.randn(500, 256).cuda()  # [seq_len, hidden_size]
    output, weights = model(_input)
    print(f"Output shape: {output.shape}\tattention shape: {weights.shape}")
    """
