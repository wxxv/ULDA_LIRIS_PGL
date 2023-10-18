import torch
import numpy as np
import torch.nn.functional as F
from config import Config
cfg = Config()


def match_weight(target, dict_sta):
    for section in dict_sta:
        if dict_sta[section]['min'] < target <= dict_sta[section]['max']:
            # print(target)
            # print(section)
            # print(dict_sta[section])
            if dict_sta[section]['num_smooth']==0:
                dict_sta[section]['num_smooth'] = 1
            return np.float32(1 / dict_sta[section]['num_smooth'])


def cal_weights(targets, dict_sta):
    weights = [match_weight(target, dict_sta) for target in targets]
    scaling = len(weights) / np.sum(weights)
    weights = [scaling * x for x in weights]
    return weights


def match_weight_smo_div_num(target, dict_sta):
    for section in dict_sta:
        if dict_sta[section]['min'] < target <= dict_sta[section]['max']:
            # print(target)
            # print(section)
            # print(dict_sta[section])
            if dict_sta[section]['num_smooth']==0:
                dict_sta[section]['num_smooth'] = 1

            ret = np.float32(dict_sta[section]['num_smooth'] / dict_sta[section]['num'])
            if ret > 1:
                ret = 1
            return ret

            # return np.float32(dict_sta[section]['num_smooth'] / dict_sta[section]['num'])


def cal_weights_smo_div_num(targets, dict_sta):
    weights = [match_weight_smo_div_num(target, dict_sta) for target in targets]
    scaling = len(weights) / np.sum(weights)
    weights = [scaling * x for x in weights]
    return weights


def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs.squeeze(0) - targets.squeeze(1)) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_mse_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = F.l1_loss(inputs, targets, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_dense_loss(inputs, targets, dw=None):
    loss = (inputs.view(-1) - targets.view(-1)) ** 2
    if dw is not None:
        weights = dw(targets.cpu().numpy())
        weights = torch.Tensor(weights).to(cfg.device)
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


if __name__ == '__main__':
    targets = []
    dict_sta = {}
    weights = [cal_weights(target, dict_sta) for target in targets]
    # weights = [1/200, 1/600, 1/100]
    scaling = 1 / np.sum(weights)
    weights = [scaling * x for x in weights]
    print('end')
