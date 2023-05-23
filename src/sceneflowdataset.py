
import os
import random
import chardet
import re

import torch.utils.data as data
import torch

import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np


class Datafactory(data.Dataset):
    def __init__(self, cfg, mode='train'):
        self.mode = mode
        self.cfg = cfg

        if mode == 'train':
            with open(self.cfg.traintxt, 'r') as f:
                lines = [line.rstrip() for line in f.readlines()]
        elif mode == 'val':
            with open(self.cfg.valtxt, 'r') as f:
                lines = [line.rstrip() for line in f.readlines()]
        else:
            assert 'check dataset mode!'
        self.id_list = [i for i in range(len(lines))]
        splits = [line.split() for line in lines]
        self.left = [x[0] for x in splits]
        self.right = [x[1] for x in splits]
        self.disp = [x[2] for x in splits]

    def __len__(self):
        return len(self.left)

    def __getitem__(self, id):
        left_path = self.cfg.sceneflow_home + self.left[id]
        right_path = self.cfg.sceneflow_home + self.right[id]
        disp_path = self.cfg.sceneflow_home + self.disp[id]
        left_img, right_img = cv2.imread(left_path), cv2.imread(right_path)

        dataL, scaleL = readPFM(disp_path)
        dataL = np.ascontiguousarray(dataL, dtype=np.float32)

        if self.mode == 'train':                        
            h, w, _ = left_img.shape
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img[y1:y1 + th, x1:x1 + tw, :]
            right_img = right_img[y1:y1 + th, x1:x1 + tw, :]
            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
            left_img = (left_img.astype(np.float32) / 255.)
            right_img = (right_img.astype(np.float32) / 255.)

            left_img = left_img * 2 - 1
            left_img = left_img.transpose(2, 0, 1)
            left_img = torch.from_numpy(left_img)

            right_img = right_img * 2 - 1
            right_img = right_img.transpose(2, 0, 1)
            right_img = torch.from_numpy(right_img)
        elif self.mode == 'val':
            h, w, _ = left_img.shape
            th, tw = 512, 960

            left_img = left_img[h - th:h, w - th:w, :]
            right_img = right_img[h - th:h, w - th:w, :]
            dataL = dataL[h - th:h, w - th:w]

            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            left_img = (left_img.astype(np.float32) / 255.)
            left_img = left_img * 2 - 1
            left_img = left_img.transpose(2, 0, 1)
            left_img = torch.from_numpy(left_img)

            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
            right_img = (right_img.astype(np.float32) / 255.)
            right_img = right_img * 2 - 1
            right_img = right_img.transpose(2, 0, 1)
            right_img = torch.from_numpy(right_img)
        return left_img, right_img, dataL, left_path


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    encode_type = chardet.detect(header)  
    header = header.decode(encode_type['encoding'])
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode(encode_type['encoding']))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip().decode(encode_type['encoding']))
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale
