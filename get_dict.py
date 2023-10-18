import numpy as np
import os
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt
from random import sample
import random
import torch.nn.functional as F


def init_dict(lower_limit, upper_limit, num_split):
    '''
    初始化统计标签区间样本数量的字典
    :param lower_limit: 下界
    :param upper_limit: 上界
    :param num_split: 划分区间数量
    :return: 初始化字典
    '''
    dict_origin = {}
    for i in range(num_split):
        length = (upper_limit - lower_limit) / num_split
        # 以上界为标签范围区间命名
        upper_bound = lower_limit + (i + 1) * length
        dict_origin[upper_bound] = {'num': 0, 'max': upper_bound, 'min': upper_bound - length, 'x': [], 'y': []}
        if i == 0:
            dict_origin[upper_bound]['min'] = lower_limit - 1e-5
    return dict_origin


def divide(x_axis_data, y_axis_data, dict_sta, lower_limit, upper_limit):
    '''
    统计各个区间的所有连续片段
    :param x_axis_data: 电影数据的时间轴
    :param y_axis_data: 电影数据的标签
    :param dict_sta: 输入的原始字典
    :param lower_limit: 下界
    :param upper_limit: 上界
    :return: 含有连续片段的字典
    '''
    x_region = []  # 连续片段的时间轴
    x_region_y = []  # 连续片段的标签
    region_pre = None
    i = 0
    while i != len(x_axis_data):
        for region in dict_sta:
            if y_axis_data[i] > upper_limit or y_axis_data[i] < lower_limit:
                print('error x = ', x_axis_data[i])
                print('error data = ', y_axis_data[i])
            if dict_sta[region]['min'] < y_axis_data[i] <= dict_sta[region]['max']:
                if region == region_pre or region_pre is None:
                    x_region.append(x_axis_data[i])
                    x_region_y.append(y_axis_data[i])
                    region_pre = region
                    if i == len(x_axis_data) - 1:
                        dict_sta[region_pre]['x'].append(x_region)
                        dict_sta[region_pre]['y'].append(x_region_y)
                    i = i + 1
                    dict_sta[region]['num'] += 1
                    break
                else:
                    # 遇到不连续标签
                    dict_sta[region_pre]['x'].append(x_region)
                    x_region = []
                    dict_sta[region_pre]['y'].append(x_region_y)
                    x_region_y = []
                    region_pre = None
                    break


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / sum(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(
            map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window


def smooth(dict_new, ks=7, sigma=3):
    '''
    平滑区间数量
    :param dict_new:
    :param ks:
    :param sigma:
    :return: 平滑后的dict
    '''
    section_num = []  # 平滑y
    section_name = []  # 平滑x
    for section in dict_new:
        section_name.append(section)
        section_num.append(dict_new[section]['num'])
    lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=ks, sigma=sigma)
    section_num_nonzero_index = [i for i, sect_num in enumerate(section_num) if sect_num != 0]
    # for i in range(len(section_num_nonzero_index) - 1):
    #     if section_num_nonzero_index[i + 1] - section_num_nonzero_index[i] != 1:
    #         print('Discontinuous!')
    #         print(section_num_nonzero_index)
    section_num_nonzero = [section_num[i] for i in section_num_nonzero_index]
    section_num_smooth_nonzero = convolve1d(np.array(section_num_nonzero), weights=lds_kernel_window, mode='constant')
    section_num_smooth = section_num.copy()
    for i in range(len(section_num_nonzero_index)):
        section_num_smooth[section_num_nonzero_index[i]] = section_num_smooth_nonzero[i]
    for i in range(len(section_num_smooth)):
        dict_new[section_name[i]]['num_smooth'] = section_num_smooth[i]


def get_statistics(labels, lower_limit, upper_limit, num_split, ks, sigma):
    dict_sta = init_dict(lower_limit, upper_limit, num_split)
    init_dict(lower_limit, upper_limit, num_split)
    time = [i for i in range(len(labels))]
    divide(time, labels, dict_sta, lower_limit, upper_limit)
    smooth(dict_sta, ks, sigma)
    return dict_sta


def get_movie_dict(dataset, lower_limit, upper_limit, num_split, ks, sigma, mode='valance'):
    dict_all = {}
    for id in range(len(dataset)):
        if mode == 'valence':
            labels = dataset[id][-1]
        elif mode == 'arousal':
            labels = dataset[id][-2]
        else:
            print('wrong mode!')
        dict_all[id] = get_statistics(labels, lower_limit, upper_limit, num_split, ks, sigma)
    return dict_all


if __name__ == '__main__':
    # 示例
    train_set = np.load('./va_datas/train.npy', allow_pickle=True, encoding='latin1')
    test_set = np.load('./va_datas/test.npy', allow_pickle=True, encoding='latin1')
    # 超参数设置
    lower_limit = -1
    upper_limit = 1
    num_split = 10
    ks = 0.7
    sigma = 2
    dict_all = get_movie_dict(train_set, lower_limit, upper_limit, num_split, ks, sigma, mode='valance')
    movies = 10
    for i in movies:
        dict_sta = dict_all[i]

