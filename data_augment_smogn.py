import torch
import torch.nn.functional as F
import random


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


def smoter(case1, case2, case1_target, case2_target, dist):
    p = random.random()
    new_case = p*case1+(1-p)*case2
    dist1 = torch.norm(new_case-case1)
    dist2 = torch.norm(new_case-case2)
    new_target = (dist2*case1_target + dist1*case2_target)/dist
    return new_case, new_target


def add_noise(case, t_pert):
    mean = 0
    std = case.std().item()
    noise = torch.ones(case.size()) * std + mean
    return case + noise.cuda(3) * t_pert


def smogn_sample(t, dict_all, sample_num, k, pert, t_max):
    t_len = len(t)
    if t_len < k+1:
        t, t_len = sec_extend(t, t_len, k+1, t_max)
    sec_features = torch.stack([dict_all[i]['feature'] for i in t])
    sec_labels = torch.stack([dict_all[i]['label'] for i in t])
    indices = torch.randperm(sec_features.shape[0])[:sample_num]
    for i in indices:
        case1 = sec_features[i]
        case1_target = sec_labels[i]
        mask = torch.ones(sec_features.shape[0], dtype=torch.bool)
        mask[i] = False
        sec_features_others = sec_features[mask]
        sec_labels_others = sec_labels[mask]
        distances = torch.cdist(sec_features[i].unsqueeze(dim=0), sec_features_others).view(-1)
        _, neighbours_index = distances.topk(k=k, largest=False)  # k个特征的index,对应dis和others
        maxd = distances.median()
        case2_index = neighbours_index[random.randint(0, neighbours_index.numel() - 1)]
        case2 = sec_features_others[case2_index]
        case2_targets = sec_labels_others[case2_index]
        dist_case = distances[case2_index]
        if dist_case < maxd:
            new_case, new_target = smoter(case1, case2, case1_target, case2_targets, dist_case)
        else:
            t_pert = min(maxd, pert)
            new_case = add_noise(case1, t_pert)
            new_target = case1_target

        sim = [F.cosine_similarity(new_case, sec_features[i], dim=-1) for i in range(len(sec_features))]
        max_index = sim.index(max(sim))
        t_insert = t[max_index]
        sample_label = dict_all[t_insert]['label']
        t_add = random.uniform(-0.5, 0.5)
        dict_all[t_insert + t_add] = {'feature': new_case, 'label': sample_label}


def smogn_aug(features, target, dict_new, k=5, pert=0.1):
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
                smogn_sample(x, dict_aug, sample_num, k, pert, t_max)
    features_aug = []
    labels_aug = []
    for time in sorted(dict_aug):
        features_aug.append(dict_aug[time]['feature'])
        labels_aug.append(dict_aug[time]['label'])
    features_aug = torch.stack(features_aug)
    labels_aug = torch.stack(labels_aug)

    return features_aug, labels_aug


if __name__ == '__main__':
    sec_features = torch.stack([torch.randn(128) for _ in range(10)])
    sec_labels = torch.stack([torch.randn(1) for _ in range(10)])
    sample_num = 3
    k = 5
    pert = 0.02
    indices = torch.randperm(sec_features.shape[0])[:sample_num]
    for i in indices:
        case1 = sec_features[i]
        case1_target = sec_labels[i]
        mask = torch.ones(sec_features.shape[0], dtype=torch.bool)
        mask[i] = False
        sec_features_others = sec_features[mask]
        sec_labels_others = sec_labels[mask]
        distances = torch.cdist(sec_features[i].unsqueeze(dim=0), sec_features_others).view(-1)
        _, neighbours_index = distances.topk(k=k, largest=False)  # k个特征的index,对应dis和others
        maxd = distances.median()
        case2_index = neighbours_index[random.randint(0, neighbours_index.numel() - 1)]
        case2 = sec_features_others[case2_index]
        case2_targets = sec_labels_others[case2_index]
        dist_case = distances[case2_index]
        if dist_case < maxd:
            new_case, new_target = smoter(case1, case2, case1_target, case2_targets, dist_case)
        else:
            t_pert = min(maxd, pert)
            new_case = add_noise(case1, t_pert)
            new_target = case1_target
