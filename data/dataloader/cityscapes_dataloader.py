import numpy as np
import os, json
import cv2
import random
from copy import deepcopy

import torch
from torch.utils.data import Dataset, DataLoader

class CityScapes(Dataset):
    def __init__(self, dataroot, mode, task, crop_h=None, crop_w=None):
        self.dataroot = dataroot
        self.mode = mode
        self.task = task
        
        if crop_h is not None and crop_w is not None:
            self.crop_h = crop_h
            self.crop_w = crop_w
        else:
            self.crop_h = 256
            self.crop_w = 512
        
        # IMG MEAN is in BGR order
        self.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        self.IMG_MEAN = np.tile(self.IMG_MEAN[np.newaxis, np.newaxis, :], (self.crop_h, self.crop_w, 1))
        
        json_file = 'data/utils/cityscape.json'
        with open(json_file, 'r') as f:
            info = json.load(f)
        self.files = info[self.mode]
        
#         self.files = []
#         self.labels = []
        
#         filepath = os.path.join(self.dataroot, self.mode, 'image')
#         if self.task is 'depth_zbuffer':
#             labelpath = os.path.join(self.dataroot, self.mode, 'depth')
#         elif self.task is 'segment_semantic':
#             labelpath = os.path.join(self.dataroot, self.mode, 'label_19')
#         for file in os.listdir(filepath):
#             if file.endswith('.npy'):
#                 self.files.append(os.path.join(filepath, file))
#                 self.labels.append(os.path.join(labelpath, file))
        

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
        label_new = np.expand_dims(cv2.resize(label, (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
        return img_new, label_new

    def __mirror__(self, img, label):
        flag = random.random()
        if flag > 0.5:
            img = img[:, ::-1]
            label = label[:, ::-1]
        return img, label

    def __random_crop_and_pad_image_and_labels__(self, img, label, ignore_label=-1.0):
        # combining
        label = label.astype('float32')
        if self.task is 'segment_semantic':
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

        return img_crop, label_crop
    
    def __scale__val__(self, img, label):
        img_new = cv2.resize(img, (self.crop_w, self.crop_h))
        label_new = np.expand_dims(cv2.resize(label, (self.crop_w, self.crop_h), interpolation=cv2.INTER_NEAREST), axis=-1)
        return img_new, label_new

    def __getitem__(self, idx):
        img_path, depth_path, label2_path, label7_path, label19_path = self.files[idx]
        
        img = np.load(os.path.join(self.dataroot, img_path))[:, :, ::-1] * 255
        if self.task is 'segment_semantic':
            label = np.expand_dims(np.load(os.path.join(self.dataroot, label19_path)), axis=-1)
        elif  self.task is 'depth_zbuffer':
            label = np.load(os.path.join(self.dataroot, depth_path))

        if self.mode in ['train', 'train2']:
            img, label = self.__scale__(img, label)
            img, label = self.__mirror__(img, label)
            img, label = self.__random_crop_and_pad_image_and_labels__(img, label)
#         elif self.mode is 'val':
#             img, label = self.__scale__val__(img, label)

        img = img.astype('float')
        img -= self.IMG_MEAN
        
        name = img_path.split('/')[-1]
        
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        if self.task is 'segment_semantic':
            label = torch.from_numpy(label).permute(2, 0, 1)
        elif  self.task is 'depth_zbuffer':
            label = torch.from_numpy(label).permute(2, 0, 1).float()

        return {'input': img, 'label': label, 'name': name}
#         return {'input': img, 'label': label}