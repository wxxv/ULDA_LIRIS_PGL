import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d
from random import sample
from data_loader import get_loader
# from configs import get_config
import random
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


def init_dict(n):
    dict_origin = {}
    for i in range(n):
        upper_bound = (i + 1) * 1 / n
        dict_origin[upper_bound] = {'num': 0, 'max': (i + 1) * 1 / n, 'min': i * 1 / n, 'x': [], 'y': []}
        if i == 0:
            dict_origin[upper_bound]['min'] = -0.1
    return dict_origin


def divide(x_axis_data, y_axis_data, dict_new):
    x_region = []
    x_region_y = []
    region_pre = None
    i = 0
    while i != len(x_axis_data):
        for region in dict_new:
            if y_axis_data[i] > 1 or y_axis_data[i] < 0:
                print('i = ', x_axis_data[i])
                print('error data = ', y_axis_data[i])
            if dict_new[region]['min'] < y_axis_data[i] <= dict_new[region]['max']:
                if region == region_pre or region_pre is None:
                    x_region.append(x_axis_data[i])
                    x_region_y.append(y_axis_data[i])
                    region_pre = region
                    if i == len(x_axis_data) - 1:
                        dict_new[region_pre]['x'].append(x_region)
                        dict_new[region_pre]['y'].append(x_region_y)
                    i = i + 1
                    dict_new[region]['num'] += 1
                    break
                else:
                    dict_new[region_pre]['x'].append(x_region)
                    x_region = []
                    dict_new[region_pre]['y'].append(x_region_y)
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


def smooth(dict_new):
    section_num = []
    section_name = []
    for section in dict_new:
        section_name.append(section)
        section_num.append(dict_new[section]['num'])
    lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=3, sigma=2)
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


def insert(index, t_data, num, movie_smooth):
    if num == 0:
        return 0
    for n in range(num):
        p = np.random.beta(0.5, 0.5)
        if p < 0.5:
            p = 1 - p
        t = p * t_data[index] + (1 - p) * (t_data[index] + 1)
        insert_feature = p * movie_smooth[t_data[index]]['feature'] + (1 - p) * movie_smooth[t_data[index] + 1][
            'feature']
        insert_label = p * movie_smooth[t_data[index]]['label'] + (1 - p) * movie_smooth[t_data[index] + 1]['label']
        movie_smooth[t] = {'feature': insert_feature, 'label': insert_label}


def down_sample(t_data, movie_smooth, num_keep):
    t_keep = sample(t_data, num_keep)
    for t in t_keep:
        movie_smooth.pop(t)


def allocate(dict_new):
    down = False
    for section in dict_new:
        num_smooth = dict_new[section]['num_smooth']
        num_raw = dict_new[section]['num']
        sec_num = len(dict_new[section]['x'])
        if num_smooth > num_raw:
            sec_len = [len(x_sec) for x_sec in dict_new[section]['x']]
            sec_pick = random.choices([i for i in range(sec_num)], weights=sec_len, k=num_smooth - num_raw)
            sample_num = [0 for _ in range(sec_num)]
            for pick in sec_pick:
                sample_num[pick] += 1
            dict_new[section]['sample_num'] = sample_num
        else:
            dict_new[section]['sample_num'] = 0


def sec_extend(t, t_len, min_len, t_max):
    i = 0
    while t_len < min_len:
        if t[t_len - 1] != t_max-1 and i % 2 == 0:
            t.append(t[t_len - 1] + 1)
            i += 1
            t_len += 1
        elif t[0] != 0 and i % 2 == 1:
            t.insert(0, t[0] - 1)
            i += 1
            t_len += 1
        else:
            i += 1
    return t, t_len


def gaussian_sample(t, dict_all, sample_num, t_max):
    min_len = 3
    t_len = len(t)
    if t_len < min_len:
        t, t_len = sec_extend(t, t_len, min_len, t_max)
    sec_features = torch.stack([dict_all[i]['feature'] for i in t])
    mean = torch.mean(sec_features, dim=0).cpu().detach().numpy()
    cov = torch.cov(sec_features.T).cpu().detach().numpy()
    # dis = MultivariateNormal(mean, cov)
    for i in range(sample_num):
        # sample_feature = dis.sample()
        sample_feature = torch.tensor(np.random.multivariate_normal(mean, cov), dtype=torch.float32).to("cuda:1")
        # check_feature = sample_feature.cpu().numpy()
        sim = [F.cosine_similarity(sample_feature, sec_features[i], dim=-1) for i in range(t_len)]
        max_index = sim.index(max(sim))
        t_insert = t[max_index]
        sample_label = dict_all[t_insert]['label']
        t_add = random.uniform(-0.5, 0.5)
        dict_all[t_insert + t_add] = {'feature': sample_feature, 'label': sample_label}


def gaussian_aug(features, target, dict_new):
    """
    1.分配每个区间要采样的数量
    """
    allocate(dict_new)
    t_max = len(target)
    time = [i for i in range(t_max)]
    dict_aug = {time[i]: {'feature': features[i], 'label': target[i]} for i in range(t_max)}
    for section in dict_new:
        if dict_new[section]['sample_num'] != 0:
            for x, sample_num in zip(dict_new[section]['x'], dict_new[section]['sample_num']):
                gaussian_sample(x, dict_aug, sample_num, t_max)
    features_aug = []
    labels_aug = []
    for time in sorted(dict_aug):
        features_aug.append(dict_aug[time]['feature'])
        labels_aug.append(dict_aug[time]['label'])
    features_aug = torch.stack(features_aug)
    labels_aug = torch.stack(labels_aug)

    return features_aug, labels_aug


def data_augment(features, target):
    n = 10
    dict_new = init_dict(n)
    time = [i for i in range(len(features))]
    divide(time, target, dict_new)
    smooth(dict_new)
    features_aug, target_aug = gaussian_aug(features, target, dict_new)
    return features_aug, target_aug


if __name__ == '__main__':
    random.seed(12345)
    config = get_config(mode='train')
    train_loader = get_loader(config.mode, config.video_type, 3)
    lenth = len(train_loader)
    iterator = iter(train_loader)
    for _ in range(lenth):
        frame_features, target = next(iterator)
        frame_features = frame_features.squeeze(0)
        target = target.squeeze(0)
        frame_features = frame_features.to(config.device)
        # vector = torch.ones(frame_features.size()).to(config.device)
        # frame_features = frame_features + vector
        target = target.to(config.device)
        features_aug, target_aug = data_augment(frame_features, target)
