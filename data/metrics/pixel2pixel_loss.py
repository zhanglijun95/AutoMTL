import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class DataCriterions(nn.Module):
    def __init__(self, task):
        super(DataCriterions, self).__init__()
        self.task = task
        
    def define_loss(self):
        self.cosine_similiarity = nn.CosineSimilarity()
        self.l1_loss = nn.L1Loss()
        self.l1_loss_sum = nn.L1Loss(reduction='sum')
        
    def seg_loss(self, pred, gt):
        prediction = pred.permute(0, 2, 3, 1).contiguous().view(-1, self.num_seg_cls)
        new_shape = pred.shape[-2:]
        gt = F.interpolate(gt.float(), size=new_shape).permute(0, 2, 3, 1).contiguous().view(-1)
        loss = self.cross_entropy(prediction, gt.long())
        return loss
    
    def sn_loss(self, pred, gt, mask=None):
        prediction = pred.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        new_shape = pred.shape[-2:]
        gt = F.interpolate(gt.float(), size=new_shape).permute(0, 2, 3, 1).contiguous().view(-1, 3)
        labels = (gt.max(dim=1)[0] < 255)
        if mask is not None:
            gt_mask = F.interpolate(mask.float(), size=new_shape).permute(0, 2, 3, 1).contiguous().view(-1, 3)
            labels = torch.logical_and(labels, gt_mask[:, 0].int() == 1)
        prediction = F.normalize(prediction[labels])
        gt = F.normalize(gt[labels])
        loss = 1 - self.cosine_similiarity(prediction, gt).mean()
        return loss
    
    def depth_loss(self, pred, gt, mask=None):
        new_shape = pred.shape[-2:]
        gt = F.interpolate(gt.float(), size=new_shape) 
        if mask is not None:
            gt_mask = F.interpolate(mask.float(), size=new_shape)
            binary_mask = (gt != 255) * (gt_mask.int() == 1)
        else:
            binary_mask = (torch.sum(gt, dim=1) > 3 * 1e-5).unsqueeze(1)
        prediction = pred.masked_select(binary_mask)
        gt = gt.masked_select(binary_mask)
        loss = self.l1_loss(prediction, gt)
#         loss = self.l1_loss_sum(prediction, gt)
        return loss
    
    def keypoint_edge_loss(self, pred, gt):
        new_shape = pred.shape[-2:]
        gt = F.interpolate(gt.float(), size=new_shape)
        binary_mask = gt != 255
        prediction = pred.masked_select(binary_mask)
        gt = gt.masked_select(binary_mask)
        loss = self.l1_loss(prediction, gt)
        return loss
        
    def forward(self, pred, gt, mask=None):
        if self.task == 'segment_semantic':
            return self.seg_loss(pred, gt)
        elif self.task == 'normal':
            return self.sn_loss(pred, gt, mask)
        elif self.task == 'depth_zbuffer':
            return self.depth_loss(pred, gt, mask)
        elif self.task == 'keypoints2d' or self.task == 'edge_texture':
            return self.keypoint_edge_loss(pred, gt)
        
class TaskonomyCriterions(DataCriterions):
    def __init__(self, task, dataroot):
        super(TaskonomyCriterions, self).__init__(task)
        if self.task == 'segment_semantic':
            self.num_seg_cls = 17
        self.define_loss(dataroot)
        
    def define_loss(self, dataroot):
        super(TaskonomyCriterions, self).define_loss()
        weight = torch.from_numpy(np.load(os.path.join(dataroot, 'semseg_prior_factor.npy'))).float()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, ignore_index=255)
        
class CityScapesCriterions(DataCriterions):
    def __init__(self, task):
        super(CityScapesCriterions, self).__init__(task)
        if self.task == 'segment_semantic':
            self.num_seg_cls = 19
        self.define_loss()
        
    def define_loss(self):
        super(CityScapesCriterions, self).define_loss()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-1)
        
class NYUCriterions(DataCriterions):
    def __init__(self, task):
        super(NYUCriterions, self).__init__(task)
        if self.task == 'segment_semantic':
            self.num_seg_cls = 40
        self.define_loss()
        
    def define_loss(self):
        super(NYUCriterions, self).define_loss()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=255)