import numpy as np
import os
import cv2
import random
import json
from copy import deepcopy

import torch
from torch.utils.data import Dataset, DataLoader


class NYU_v2(torch.utils.data.Dataset):
    def __init__(self, dataroot, mode, task, crop_h=None, crop_w=None):
        self.dataroot = dataroot
        self.mode = mode
        self.task = task
        
        if crop_h is not None and crop_w is not None:
            self.crop_h = crop_h
            self.crop_w = crop_w
        else:
            self.crop_h = 480
            self.crop_w = 640
        
        # IMG MEAN is in BGR order
        self.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        self.IMG_MEAN = np.tile(self.IMG_MEAN[np.newaxis, np.newaxis, :], (self.crop_h, self.crop_w, 1))
        
        json_file = dataroot + 'nyu_v2_3task.json'
        with open(json_file, 'r') as f:
            info = json.load(f)
        self.files = info[self.mode]
        

    def __len__(self):
        return len(self.files)

    def __scale__(self, img, label):
        """
           Randomly scales the images between 0.5 to 1.5 times the original size.
        """
        # random value between 0.5 and 1.5
        scale = random.random() + 0.5
        h, w, _ = img.shape
        h_new = int(h * scale)
        w_new = int(w * scale)
        img_new = cv2.resize(img, (w_new, h_new))
        if self.task is 'normal':
            label_new = cv2.resize(label, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
        else:
            label_new = np.expand_dims(cv2.resize(label, (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)

        return img_new, label_new

    def __mirror__(self, img, label):
        flag = random.random()
        if flag > 0.5:
            img = img[:, ::-1]
            label = label[:, ::-1]
        return img, label

    def __random_crop_and_pad_image_and_labels__(self, img, label, ignore_label=255):
        # combining
        label = label.astype('float32')
        if self.task is 'segment_semantic' or self.task is 'normal':
            label -= ignore_label
        combined = np.concatenate((img, label), axis=2)
        image_shape = img.shape
        
        # padding to the crop size
        pad_shape = [max(image_shape[0], self.crop_h), max(image_shape[1], self.crop_w), combined.shape[-1]]
        combined_pad = np.zeros(pad_shape)
        offset_h, offset_w = (pad_shape[0] - image_shape[0])//2, (pad_shape[1] - image_shape[1])//2
        combined_pad[offset_h: offset_h+image_shape[0], offset_w: offset_w+image_shape[1]] = combined
        
        # cropping
        crop_offset_h, crop_offset_w = pad_shape[0] - self.crop_h, pad_shape[1] - self.crop_w
        start_h, start_w = np.random.randint(0, crop_offset_h+1), np.random.randint(0, crop_offset_w+1)
        combined_crop = combined_pad[start_h: start_h+self.crop_h, start_w: start_w+self.crop_w]
        
        # separating
        img_cdim = image_shape[-1]
        img_crop = deepcopy(combined_crop[:, :, :img_cdim])
        label_crop = deepcopy(combined_crop[:, :, img_cdim:]).astype('float')
        if self.task is 'segment_semantic':
            label_crop = (label_crop + ignore_label).astype('int')
        elif self.task is 'normal':
            label_crop = (label_crop + ignore_label)
        return img_crop, label_crop

    def __scale__val__(self, img, label):
        img_new = cv2.resize(img, (self.crop_w, self.crop_h))
        if self.task is 'normal':
            label_new = cv2.resize(label, (self.crop_w, self.crop_h), interpolation=cv2.INTER_NEAREST)
        else:
            label_new = np.expand_dims(cv2.resize(label, (self.crop_w, self.crop_h), interpolation=cv2.INTER_NEAREST), axis=-1)
        return img_new, label_new
    
    def __getitem__(self, idx):
        img_path, seg_path, normal_path, depth_path = self.files[idx]
        img = cv2.imread(os.path.join(self.dataroot, img_path))
        if self.task is 'segment_semantic':
            label = np.expand_dims(cv2.imread(os.path.join(self.dataroot, seg_path), cv2.IMREAD_GRAYSCALE), axis=-1)
        elif self.task is 'normal':
            label = cv2.imread(os.path.join(self.dataroot, normal_path))
        elif self.task is 'depth_zbuffer':
            label = np.expand_dims(np.load(os.path.join(self.dataroot, depth_path)), axis=-1)
            
        if self.mode is 'train':
            img, label = self.__scale__(img, label)
            img, label = self.__mirror__(img, label)
            img, label = self.__random_crop_and_pad_image_and_labels__(img, label)
        elif self.mode in ['test', 'val']:
            img, label = self.__scale__val__(img, label)

        img = img.astype('float')
        img -= self.IMG_MEAN
        
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        if self.task is 'segment_semantic':
            label =  torch.from_numpy(label).permute(2, 0, 1)
        else:
            label =  torch.from_numpy(label).permute(2, 0, 1).float()

        return {'input': img, 'label': label}