from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from torch import optim
from config import Config
from loss import *
from get_dict import *

import datetime
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from utils import obtain_samples, moving_mean
from layers.summarizer import PGL_SUM
# from layers.summarizer_transformer import PGL_SUM
import pickle
from tqdm import tqdm, trange
from denseweight import DenseWeight
cfg = Config()
print(cfg)

# with open('./kernel_ablation_result/valence_T15.txt', 'a') as f:
#     f.write(str(cfg))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(2000)


device = cfg.device


def train():
    model = PGL_SUM(pos_enc=cfg.pos_enc, heads=cfg.heads, num_segments=cfg.num_segments,
                    fusion=cfg.fusion).to(device)
    # model.load_state_dict(torch.load('./pre_training/v_backbone/best_valence_090_429.pth'))
    aug = cfg.aug
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    best_test_loss = 0.0
    best_test_pcc = 0.0
    best_epoch = 0
    mse = []
    pcc = []
    train_data_list = np.load(cfg.train_path, allow_pickle=True, encoding='latin1')
    dict_all = get_movie_dict(train_data_list, -1, 1, 100, ks=3, sigma=1, mode=cfg.task)
    weights_list = []

    for epoch in trange(cfg.num_epochs, desc='Epoch', ncols=80, leave=False):
        # """
        torch.cuda.empty_cache()
        train_step = 0
        movies = 0
        for train_data in tqdm(train_data_list, desc='Batch', ncols=80, leave=False):
            movies += 1
            vggish, bgm_emotion, emotion, scene, pose, _, _, arousalValue, valenceValue = train_data

            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            assert vggish.shape[0] == emotion.shape[0] == bgm_emotion.shape[0] == \
                   scene.shape[0] == pose.shape[0]

            train_step += 1
            if cfg.task == 'arousal':
                labels_ori = torch.from_numpy(arousalValue).to(device).to(torch.float32)
            else:
                labels_ori = torch.from_numpy(valenceValue).to(device).to(torch.float32)
            input = []
            input.extend([torch.from_numpy(vggish).to(device).to(torch.float32),
                          torch.from_numpy(bgm_emotion).to(device).to(torch.float32),
                          torch.from_numpy(emotion).to(device).to(torch.float32),
                          torch.from_numpy(scene).to(device).to(torch.float32),
                          torch.from_numpy(pose).to(device).to(torch.float32)
                          ])

            dict_sta = dict_all[movies-1]
            output, _, targets = model(input, labels_ori, aug=aug, epoch=epoch, movies=movies, dict_sta=dict_sta)
            # print(f'movie:{movies}...len:{len(targets)}')

            # loss = criterion(output.squeeze(0), targets.squeeze(1))
            # loss = bmc_loss(output, target, torch.nn.Parameter(torch.tensor(1., device='cuda:2')))
            if epoch == 0:
                # weights = torch.Tensor(cal_weights(targets, dict_sta)).to(device).detach()    # inverse
                weights = torch.Tensor(cal_weights_smo_div_num(targets, dict_sta)).to(device).detach()    # smo / num
                # dw = DenseWeight(alpha=1.0)
                # dw.fit(targets.cpu().numpy())
                weights_list.append(weights)
                # print(weights)
            else:
                weights = weights_list[movies-1]

            loss = weighted_mse_loss(output, targets, weights)
            # loss = weighted_focal_mse_loss(output, targets, None)
            # loss = weighted_dense_loss(output, targets, dw)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

        # """

        test_step = 0
        test_mse = 0.0
        test_pcc = 0.0
        model.eval()
        with torch.no_grad():
            test_data_list = np.load(cfg.test_path, allow_pickle=True, encoding='latin1')

            gt_labels_list = []
            pre_labels_list = []
            for test_data in tqdm(test_data_list, desc='Batch', ncols=80, leave=False):
                vggish, bgm_emotion, emotion, scene, pose, _, _, arousalValue, valenceValue = test_data
                assert vggish.shape[0] == emotion.shape[0] == bgm_emotion.shape[0] == \
                       scene.shape[0] == pose.shape[0]
                test_step += 1
                if cfg.task == 'arousal':
                    labels_ori = torch.from_numpy(arousalValue).to(device).to(torch.float32)
                else:
                    labels_ori = torch.from_numpy(valenceValue).to(device).to(torch.float32)
                input = []
                input.extend([torch.from_numpy(vggish).to(device).to(torch.float32),
                              torch.from_numpy(bgm_emotion).to(device).to(torch.float32),
                              torch.from_numpy(emotion).to(device).to(torch.float32),
                              torch.from_numpy(scene).to(device).to(torch.float32),
                              torch.from_numpy(pose).to(device).to(torch.float32)
                              ])

                pre_labels = []

                pre_labels, _, target = model(input, labels_ori, aug=False, epoch=epoch, movies=None, dict_sta=None)
                pre_labels = pre_labels.squeeze(0).cpu()
                target = target.squeeze(1).cpu()

                test_pcc += pearsonr(pre_labels, target)[0]
                test_mse += np.mean(np.square(np.array(target - pre_labels)))

                gt_labels_list.append(target)
                pre_labels_list.append(pre_labels)
            test_pcc = test_pcc / test_step
            test_mse = test_mse / test_step
            # scheduler.step(test_mse)
            pcc.append(test_pcc)
            mse.append(test_mse)
            if test_pcc > best_test_pcc:
                best_test_loss = test_mse
                best_test_pcc = test_pcc
                best_epoch = epoch
                dic = {}
                dic['gt_labels_list'] = gt_labels_list
                dic['pre_labels_list'] = pre_labels_list
                # with open(f'./result/gt_pre_{cfg.task}_backbone.pkl', 'wb') as fo:
                #     pickle.dump(dic, fo, protocol=4)
                #     fo.close()
                #     tqdm.write('Save gt_pre in result')
            # torch.save(model.state_dict(), f'./pre_training/v_swloss_mc/epoch_{epoch}_{cfg.task}_{test_mse}_{test_pcc}.pth')
            # tqdm.write(f'Save in ./pre_training/epoch_{epoch}_{cfg.task}.pth')

            now_time = datetime.datetime.now()
            str_time = now_time.strftime("%Y-%m-%d %H:%M:%S")
            test_log_str = '\r[%s] Epoch:%d, movies:%d test_loss:%.3f, test_pcc:%.3f' \
                           ', best_test_loss:%.3f, best_test_pcc:%.3f, best_epoch:%d' \
                           % (str_time, epoch, test_step, test_mse, test_pcc,
                              best_test_loss, best_test_pcc, best_epoch)
            tqdm.write(test_log_str)
            # with open('./kernel_ablation_result/valence_T15.txt', 'a') as f:
            #     f.write(test_log_str)
            # torch.save(model.state_dict(), f'./pre_training/epoch{epoch}_pcc{test_pcc}_mse{test_mse}.pt')
            # print(f'Save ./pre_training/epoch{epoch}_pcc{test_pcc}_mse{test_mse}.pt')



if __name__ == "__main__":
    train()
