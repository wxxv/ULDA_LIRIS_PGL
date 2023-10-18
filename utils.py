# -*- coding: UTF-8 -*-

"""
@author: zhangjie
@software: Pycharm
@license: Apache Licence
@file: utils.py
@time: 2022/10/8 2:54 下午
@desc:
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import torch


def obtain_samples(vggish, bgm_emotion, emotion, scene, pose, labels, time_steps, device):
    total_seconds_num = vggish.shape[0]
    start = 0
    samples = []
    while start < total_seconds_num:
        stop = start + time_steps
        if stop > total_seconds_num:
            break

        sample = []
        torch_labels = torch.reshape(torch.from_numpy(labels[start:stop]).to(device), (-1, time_steps)).mean(dim=-1, keepdim=True)
        sample.extend([torch.from_numpy(vggish[start:stop]).unsqueeze(dim=0).to(device).to(torch.float32),
                       torch.from_numpy(bgm_emotion[start:stop]).unsqueeze(dim=0).to(device).to(torch.float32),
                       torch.from_numpy(emotion[start:stop]).unsqueeze(dim=0).to(device).to(torch.float32),
                       torch.from_numpy(scene[start:stop]).unsqueeze(dim=0).to(device).to(torch.float32),
                       torch.from_numpy(pose[start:stop]).unsqueeze(dim=0).to(device).to(torch.float32),
                       torch_labels])
        samples.append(sample)
        start = stop
    return samples


def moving_mean(data, momentum=0.97):
    """
    moving_mean used in arousal postprocessing
    :param data: predicted scores of a movie
    :param momentum: momentum value
    :return:
    """
    cur_sum = 0.0
    for i in range(data.shape[0]):
        cur_sum = momentum * cur_sum + (1.0 - momentum) * data[i]
        data[i] = cur_sum
    return data