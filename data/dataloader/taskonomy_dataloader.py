import numpy as np
import os
import cv2
import random
import csv
import pickle
from PIL import Image
import json
from copy import deepcopy

import torch
from torch.utils.data import Dataset, DataLoader

class Taskonomy(Dataset):
    def __init__(self, dataroot, mode, task, crop_h=None, crop_w=None):
        self.mode = mode
        self.task = task
        
        if self.mode is 'train':
            self.dataroot = os.path.join(dataroot, self.mode)
        elif self.mode in ['test', 'test_small']:
            self.dataroot = os.path.join(dataroot, 'test')
        
        if crop_h is not None and crop_w is not None:
            self.crop_h = crop_h
            self.crop_w = crop_w
        else:
            self.crop_h = 256
            self.crop_w = 256
            
        # IMG MEAN is in BGR order
        self.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        self.IMG_MEAN = np.tile(self.IMG_MEAN[np.newaxis, np.newaxis, :], (self.crop_h, self.crop_w, 1))
        self.prior_factor = np.load(dataroot + 'semseg_prior_factor.npy')
        
        json_file = dataroot + 'taskonomy.json'
        with open(json_file, 'r') as f:
            info = json.load(f)
        self.files = info[self.mode]
        
#         # Read the split scenes' name
#         scenes = []
#         with open(self.utils_path + 'train_val_test_tiny.csv', newline='') as f:
#             reader = csv.DictReader(f)
#             for row in reader:
#                 if row[self.mode] is '1':
#                     scenes.append(row['id'])
                    
#         # Read the exceptions' name
#         exp_name = []
#         if self.mode is 'val':
#             with open(self.utils_path + 'val_exception_rgb.txt', 'rb') as fp:   
#                 exp_name = pickle.load(fp)
        
#         self.files = []
#         self.labels = []
#         for scene in scenes:
#             filepath = os.path.join(self.dataroot, self.mode, scene, 'rgb')
#             labelpath = os.path.join(self.dataroot, self.mode, scene, self.task)
#             for file in os.listdir(filepath):
#                 if file.endswith('.png'):
#                     # Don't read files in exceptions
#                     if os.path.join(scene, 'rgb', file) in exp_name:
#                         continue
                    
#                     # Modify file name for task
#                     delimiter = '_'
#                     temp = file.split(delimiter)
#                     if self.task is not 'segment_semantic':
#                         temp[-1] = self.task + '.png'
#                     else:
#                         temp[-1] =  'segmentsemantic.png'
#                     labelfile = delimiter.join(temp)
                    
#                     self.files.append(os.path.join(filepath, file))
#                     self.labels.append(os.path.join(labelpath, labelfile))
                
    def __len__(self):
        return len(self.files)
    
    def __scale__(self, img_p, label_p, label_mask = None):
        """
           Randomly scales the images between 0.5 to 1.5 times the original size.
        """
        # random value between 0.5 and 1.5
        scale = random.random() + 0.5 
        h, w, _ = img_p.shape
        h_new = int(h * scale)
        w_new = int(w * scale)
        img_new = cv2.resize(img_p, (w_new, h_new))
        
        if self.task is not 'normal':
            label_p = np.expand_dims(cv2.resize(label_p, (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
        else:
            label_p = cv2.resize(label_p, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
        
        if self.task is 'segment_semantic' or self.task is 'depth_zbuffer' or self.task is 'normal':
            label_mask = np.expand_dims(cv2.resize(label_mask, (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
        
        return img_new, label_p, label_mask

    def __mirror__(self, img_p, label_p, label_mask = None):
        flag = random.random()
        if flag > 0.5:
            img_p = img_p[:, ::-1]
            label_p = label_p[:, ::-1]
            
            if self.task is 'segment_semantic' or self.task is 'depth_zbuffer' or self.task is 'normal': 
                label_mask = label_mask[:, ::-1]
            if self.task is 'normal':
                label_p[:, :, 0] *= -1

        return img_p, label_p, label_mask

    def __random_crop_and_pad_image_and_labels__(self, img_p, label_p, label_mask = None, ignore_label=255):
        # check the ignoring labels
        if self.task is 'segment_semantic' or self.task is 'depth_zbuffer' or self.task is 'normal': 
            label = np.concatenate((label_p, label_mask), axis=2).astype('float32')
        else:
            label = label_p.astype('float32')
        label -= ignore_label
        combined = np.concatenate((img_p, label), axis=2)
        image_shape = img_p.shape
        
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
        img_crop = deepcopy(combined_crop[:, :, 0: 3])

        combined_crop[:, :, 3:] += ignore_label
        if self.task is 'normal':
            label_crop = deepcopy(combined_crop[:, :, 3: 6])
        else:
            label_crop = deepcopy(combined_crop[:, :, 3: 4])
        if self.task is 'segment_semantic' or self.task is 'depth_zbuffer':
            label_mask_crop = deepcopy(combined_crop[:, :, 4:5])
        elif self.task is 'normal':
            label_mask_crop = deepcopy(combined_crop[:, :, 6:7])
        else:
            label_mask_crop = None

        return img_crop, label_crop, label_mask_crop
    
    def semantic_segment_rebalanced(self, img, new_dims=(256, 256)):
        '''
        Segmentation
        Returns:
        --------
            pixels: size num_pixels x 3 numpy array
        '''
        img = cv2.resize(img, new_dims, interpolation=cv2.INTER_NEAREST)
        mask = img > 0.1
        mask = mask.astype(float)
        img[img == 0] = 1
        img = img - 1
        rebalance = self.prior_factor[img]
        mask = mask * rebalance
        return img, mask
    
    def rescale_image(self, img, new_scale=(-1., 1.), current_scale=None, no_clip=False):
        """
        Rescales an image pixel values to target_scale
        Args:
            img: A np.float_32 array, assumed between [0,1]
            new_scale: [min,max]
            current_scale: If not supplied, it is assumed to be in:
                [0, 1]: if dtype=float
                [0, 2^16]: if dtype=uint
                [0, 255]: if dtype=ubyte
        Returns:
            rescaled_image
        """
        img = img.astype('float32')
        # min_val, max_val = img.min(), img.max()
        # img = (img - min_val)/(max_val-min_val)
        if current_scale is not None:
            min_val, max_val = current_scale
            if not no_clip:
                img = np.clip(img, min_val, max_val)
            img = img - min_val
            img /= (max_val - min_val)
        min_val, max_val = new_scale
        img *= (max_val - min_val)
        img += min_val

        return img
    
    def resize_rescale_image(self, img, new_scale=(-1, 1), new_dims=(256, 256), no_clip=False, current_scale=None):
        """
        Resize an image array with interpolation, and rescale to be
          between
        Parameters
        ----------
        im : (H x W x K) ndarray
        new_dims : (height, width) tuple of new dimensions.
        new_scale : (min, max) tuple of new scale.
        interp_order : interpolation order, default is linear.
        Returns
        -------
        im : resized ndarray with shape (new_dims[0], new_dims[1], K)
        """
        img = img.astype('float32')
        img = cv2.resize(img, new_dims, interpolation=cv2.INTER_NEAREST)
        img = self.rescale_image(img, new_scale, current_scale=current_scale, no_clip=no_clip)
        return img
    
    def resize_and_rescale_image_log(self, img, new_dims=(256, 256), offset=1., normalizer=np.log(2. ** 16)):
        """
            Resizes and rescales an img to log-linear
            Args:
                img: A np array
                offset: Shifts values by offset before taking log. Prevents
                    taking the log of a negative number
                normalizer: divide by the normalizing factor after taking log
            Returns:
                rescaled_image
        """
        img = np.log(float(offset) + img) / normalizer
        img = cv2.resize(img, new_dims, interpolation=cv2.INTER_NEAREST)
        return img
    
    def mask_if_channel_ge(self, img, threshold, channel_idx, broadcast_to_shape=None, broadcast_to_dim=None):
        '''
            Returns a mask that masks an entire pixel iff the channel
                specified has values ge a specified value
        '''
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        h, w, c = img.shape
        mask = (img[:, :, channel_idx] < threshold)  # keep if lt
        if len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis].astype(np.float32)
        if broadcast_to_shape is not None:
            return np.broadcast_to(mask, broadcast_to_shape)
        elif broadcast_to_dim is not None:
            return np.broadcast_to(mask, [h, w, broadcast_to_dim])
        else:
            return np.broadcast_to(mask, img.shape)

    def make_depth_mask(self, img, new_dims=(256, 256), broadcast_to_dim=1):
        target_mask = self.mask_if_channel_ge(img, threshold=64500, channel_idx=0, broadcast_to_dim=broadcast_to_dim)
        target_mask = cv2.resize(target_mask, new_dims, interpolation=cv2.INTER_NEAREST)
        target_mask[target_mask < 0.99] = 0.
        return target_mask
    
    def __scale__val__(self, img_p, label_p, label_mask = None):
        img_new = cv2.resize(img_p, (self.crop_w, self.crop_h))
        
        if self.task is not 'normal':
            label_p = np.expand_dims(cv2.resize(label_p, (self.crop_w, self.crop_h), interpolation=cv2.INTER_NEAREST), axis=-1)
        else:
            label_p = cv2.resize(label_p, (self.crop_w, self.crop_h), interpolation=cv2.INTER_NEAREST)
        
        if self.task is 'segment_semantic' or self.task is 'depth_zbuffer' or self.task is 'normal':
            label_mask = np.expand_dims(cv2.resize(label_mask, (self.crop_w, self.crop_h), interpolation=cv2.INTER_NEAREST), axis=-1)
        
        return img_new, label_p, label_mask
    
    def __getitem__(self, idx):
        while True:
            img_path, seg_path, sn_path, depth_path, keypoint_path, edge_path = self.files[idx]
            try:
                # Step 1: Get input
                img = np.array(Image.open(os.path.join(self.dataroot, img_path))).astype('float32')[:, :, ::-1]
                # Step 2: Get task label
                if self.task is 'segment_semantic':
                    label_path = os.path.join(self.dataroot, seg_path)
                elif self.task is 'normal':
                    label_path = os.path.join(self.dataroot, sn_path)
                elif self.task is 'depth_zbuffer':
                    label_path = os.path.join(self.dataroot, depth_path)
                elif self.task is 'keypoints2d':
                    label_path = os.path.join(self.dataroot, keypoint_path)
                elif self.task is 'edge_texture':
                    label_path = os.path.join(self.dataroot, edge_path)
                label = np.array(Image.open(label_path))
            except:
#                 print('Error in loading %s' % self.files[idx])
                idx += 1
                if idx >= len(self.files):
                    idx = 0
            else:
                break
        
        img_p = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
        label_mask = None
        
        if self.task is 'segment_semantic':
            label_p, label_mask = self.semantic_segment_rebalanced(label)
            label_p = label_p.astype('float32')
            label_mask = label_mask.astype('float32')
            
        elif self.task is 'normal':
            label = label.astype('float32') / 255
            label_p = self.resize_rescale_image(label).astype('float32')
            
            # the mask of normal is obtained from depth mask
            depth = np.array(Image.open(os.path.join(self.dataroot, depth_path))).astype('float32')
            label_mask = self.make_depth_mask(depth).astype('float32')
        elif self.task is 'depth_zbuffer':
            label = label.astype('float32')
            label_p = self.resize_and_rescale_image_log(label)
            label_mask = self.make_depth_mask(label).astype('float32')
        elif self.task is 'keypoints2d':
            label = label.astype('float32') / (2 ** 16)
            label_p = self.resize_rescale_image(label, current_scale=(0, 0.005))
        elif self.task is 'edge_texture':
            label = label.astype('float32') / (2 ** 16)
            label_p = self.resize_rescale_image(label, current_scale=(0, 0.08))
        
        # Step 3: Transformations
        if self.mode in ['train','train1', 'train2']:
            img_p, label_p, label_mask = self.__scale__(img_p, label_p, label_mask)
            img_p, label_p, label_mask = self.__mirror__(img_p, label_p, label_mask)
            img_p, label_p, label_mask = self.__random_crop_and_pad_image_and_labels__(img_p, label_p, label_mask)
        elif self.mode in ['test', 'test_small']:
            img_p, label_p, label_mask = self.__scale__val__(img_p, label_p, label_mask)
        
        # Step 4: Append dim
        if label_p.ndim == 2:
            label_p = label_p[:, :, np.newaxis]
        if label_mask is not None:
            if label_mask.ndim == 2:
                label_mask = label_mask[:, :, np.newaxis]
                
        img_p = img_p.astype('float32')
        img_p = img_p - self.IMG_MEAN
        if self.task is 'normal':
            label_mask = np.tile(label_mask, [1, 1, 3])
                
        # Step 5: From BGR to RGB
        img_p = torch.from_numpy(img_p).permute(2, 0, 1).float()
        if self.task is 'segment_semantic':
            label_p = torch.from_numpy(label_p).permute(2, 0, 1).int()
        else:
            label_p = torch.from_numpy(label_p).permute(2, 0, 1).float()
        if label_mask is not None:
            label_mask = torch.from_numpy(label_mask).permute(2, 0, 1).float()
            return {'input': img_p, 'label': label_p, 'mask': label_mask}
        else:
            return {'input': img_p, 'label': label_p}
