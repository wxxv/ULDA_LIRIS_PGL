# -*- coding: UTF-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch


class Config(object):
    def __init__(self):
        self.hp_name = 'rl-based-mn'
        self.hp_index = '0'
        # self.task = 'arousal'
        self.task = 'valence'
        self.seed = 0
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        self.moving_mean = True
        self.moving_mean_value = 0.90

        self.num_epochs = 200
        self.show_steps = 10

        self.batch_size = 32
        self.time_steps = 10
        self.duration = 10

        # memory related
        self.memory_num = 10
        self.feature_dim = 128

        # multiple modalities
        self.modals = ['vggish', 'bgm_emotion', 'pose', 'scene', 'emotion']
        self.vggish_input_dim = 128
        self.bgm_emotion_input_dim = 128
        self.emotion_input_dim = 3072
        self.scene_input_dim = 512
        self.pose_input_dim = 128
        # self.pose_input_dim = 768

        self.modals_ouput_dim = 128

        self.lstm_layers = 2
        self.bidirectional = True

        # DDPG-related
        self.buf_num = 512
        self.train_steps = 32
        self.pi_factor = 0.3
        self.q_factor = 0.3
        self.second_pi_update = False
        self.lr = 1e-4
        self.pi_lr = 1e-4
        self.q_lr = 1e-4
        self.tao = 0.01

        # exploration
        self.noise_rate = 0.1
        self.gaussian_mean = 0.0
        self.gaussian_std = 0.05

        self.log_path = ''
        self.weights_path = ''
        self.train_path = './va_datas/train.npy'  # .npy
        self.test_path = './va_datas/test.npy'  # .npy

        self.base_model_weights_path = None
        self.noise_method = 'noise'

        # PGL_SUM
        self.pos_enc = 'absolute'
        self.heads = 8
        self.num_segments = 4
        self.fusion = 'add'
        self.aug = True
        self.weight_decay = 1e-5

    def all_parameters(self):
        hps = vars(self)
        hps_str = ''
        for key in hps.keys():
            hps_str += '{}: {}\n'.format(key, hps[key])
        return hps_str

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)