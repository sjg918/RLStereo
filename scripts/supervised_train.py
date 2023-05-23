
import logging
import datetime
import random

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from CFG.supervised_sceneflow import cfg
from src.deep_network import DCASPP_FENet, DCASPP_ANet, GHRNet
from src.sceneflowdataset import *
from src.utils import *

# nohup python supervised_train.py 1> /dev/null 2>&1 &

def train():
    # start

    # define model
    back = DCASPP_FENet().to(cfg.devices[0])
    Anet = DCASPP_ANet(cfg).to(cfg.devices[0])
    Rnet = GHRNet(cfg).to(cfg.devices[0])

    back = nn.DataParallel(back, device_ids=cfg.devices)
    Anet = nn.DataParallel(Anet, device_ids=cfg.devices)
    Rnet = nn.DataParallel(Rnet, device_ids=cfg.devices)

    # define dataloader
    sceneflow_dataset = Datafactory(cfg)
    sceneflow_loader = DataLoader(
        sceneflow_dataset, batch_size=cfg.batchsize, shuffle=True, num_workers=cfg.num_cpu,pin_memory=True, drop_last=True)

    # define optimizer and scheduler
    back_optimizer = optim.Adam(back.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999), eps=1e-08)
    back_scheduler = optim.lr_scheduler.LambdaLR(back_optimizer, cfg.burnin_schedule)
    Anet_optimizer = optim.Adam(Anet.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999), eps=1e-08)
    Anet_scheduler = optim.lr_scheduler.LambdaLR(Anet_optimizer, cfg.burnin_schedule)
    Rnet_optimizer = optim.Adam(Rnet.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999), eps=1e-08)
    Rnet_scheduler = optim.lr_scheduler.LambdaLR(Rnet_optimizer, cfg.burnin_schedule)

    back.train()
    Anet.train()
    Rnet.train()
    action_lossrecord = []
    final_lossrecord = []

    for epoch in range(1, cfg.maxepoch+1):
        # print milestones
        print('({} / {}) epoch\n'.format(epoch, cfg.maxepoch))
        with open(cfg.logdir + 'log.txt', 'a') as writer:
            writer.write('({} / {}) epoch\n'.format(epoch, cfg.maxepoch))
        
        for cnt, (left_img, right_img , dataL, left_path) in enumerate(sceneflow_loader):
            left_img, right_img, dataL = left_img.cuda(cfg.devices[0]), right_img.cuda(cfg.devices[0]), dataL.cuda(cfg.devices[0])
            B, _, H, W = left_img.shape
            mask1 = dataL < (int)(cfg.max_disp)
            mask2 = dataL > (int)(0)
            mask_big = mask1 & mask2
            mask_big.detach_()
            mask_big = dataL < (int)(cfg.max_disp)
            dataL_small = F.interpolate(dataL.unsqueeze(1), (H//4, W//4), mode='nearest').squeeze(1)
            mask1 = dataL_small < (int)(cfg.max_disp)
            mask2 = dataL_small > (int)(0)
            mask_small = mask1 & mask2
            mask_small.detach_()
            cur_disp_map = torch.zeros_like(dataL_small)
            cur_disp_map = cur_disp_map + cfg.initial_disp

            # forward
            # feature extraction
            left_fea = back(left_img)
            right_fea = back(right_img)
            
            # action iteration loop
            action_loss = 0
            for action_iteration in range(cfg.max_iteration):
                warp_fea = warp_feature_using_disparity(right_fea, cur_disp_map)
                output = Anet(left_fea, warp_fea, cur_disp_map)

                # supervise action loss
                target_action = dataL_small - cur_disp_map
                target_action = torch.clamp(target_action, -cfg.max_action, cfg.max_action)
                action_loss = action_loss + F.smooth_l1_loss(target_action[mask_small], output[mask_small], reduction='mean')

                cur_disp_map = cur_disp_map + output
                cur_disp_map = torch.clamp(cur_disp_map, 0, cfg.max_disp)
                continue

            # supervise final disp map loss
            output = Rnet(left_img, cur_disp_map)
            final_loss = F.smooth_l1_loss(dataL[mask_big], output[mask_big], reduction='mean')

            # backward
            action_loss = action_loss / cfg.max_iteration
            loss = final_loss + action_loss
            loss.backward()
            back_optimizer.step()
            back.zero_grad()
            Anet_optimizer.step()
            Anet.zero_grad()
            Rnet_optimizer.step()
            Rnet.zero_grad()

            action_lossrecord.append(action_loss.item())
            final_lossrecord.append(final_loss.item())

            # print steploss
            print("{}/{}   {}/{}   action loss: {:.2f}   final loss: {:.2f}".format(
                    epoch, cfg.maxepoch, cnt, len(sceneflow_loader),
                    sum(action_lossrecord) / len(action_lossrecord), sum(final_lossrecord) / len(final_lossrecord)), end="\r")
            continue
        
        # learning rate scheduling
        back_scheduler.step()
        Anet_scheduler.step()
        Rnet_scheduler.step()
        
        print("{}/{}   {}/{}   action loss: {:.2f}   final loss: {:.2f}".format(
                    epoch, cfg.maxepoch, cnt, len(sceneflow_loader),
                    sum(action_lossrecord) / len(action_lossrecord), sum(final_lossrecord) / len(final_lossrecord)))
        with open(cfg.logdir + 'log.txt', 'a') as writer:
                writer.write("{}/{}   {}/{}   action loss: {:.2f}   final loss: {:.2f}".format(
                    epoch, cfg.maxepoch, cnt, len(sceneflow_loader),
                    sum(action_lossrecord) / len(action_lossrecord), sum(final_lossrecord) / len(final_lossrecord)))
        action_lossrecord = []
        final_lossrecord = []

        # save model
        if epoch % 1 == 0:
            torch.save(back.state_dict(), cfg.logdir + 'back_' + str(epoch) + '.pth')
            torch.save(Anet.state_dict(), cfg.logdir + 'Anet_' + str(epoch) + '.pth')
            torch.save(Rnet.state_dict(), cfg.logdir + 'Rnet_' + str(epoch) + '.pth')
            with open(cfg.logdir + 'log.txt', 'a') as writer:
                writer.write('{} epoch model saved !\n'.format(epoch))

        continue
    # end.

if __name__ == '__main__':
    torch.manual_seed(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(777)
    np.random.seed(777)

    if os.path.exists(cfg.logdir):
       pass
    else:
        os.makedirs(cfg.logdir)

    with open(cfg.logdir + 'log.txt', 'w') as writer:
       writer.write("-start-\t")
       writer.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
       writer.write('\n\n')

    print("\n-start- ", "(", datetime.datetime.now(), ")")
    
    torch.multiprocessing.set_start_method('spawn')
    train()

    print("\n-end- ", "(", datetime.datetime.now(), ")")

    with open(cfg.logdir + 'log.txt', 'a') as writer:
        writer.write('-end-\t')
        writer.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
