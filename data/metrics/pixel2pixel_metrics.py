import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import confusion_matrix
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class DataMetrics(object):
    def __init__(self, task):
        super(DataMetrics, self).__init__()
        self.task = task
        
    def define_loss(self):
        self.cosine_similiarity = nn.CosineSimilarity()
        self.l1_loss = nn.L1Loss()
        
    def reset_records(self):
        self.batch_size = []
        if self.task is 'segment_semantic':
            self.records = {'mIoUs': [], 'pixelAccs': [],  'errs': [], 'conf_mat': np.zeros((self.num_seg_cls, self.num_seg_cls)), 'labels': np.arange(self.num_seg_cls)}
        elif self.task is 'normal':
            self.records = {'cos_similaritys': []}
        elif self.task is 'depth_zbuffer':
            self.records = {'abs_errs': [], 'rel_errs': [], 'sq_rel_errs': [], 'ratios': [], 'rms': [], 'rms_log': []}
        elif self.task is 'keypoints2d' or self.task is 'edge_texture':
            self.records = {'errs': []}
        
    def resize_pred(self, pred, gt):
        return F.interpolate(pred, size=gt.shape[-2:])
    
    def __seg_error(self, pred, gt):
        output = self.resize_pred(pred, gt)
        
        gt = gt.view(-1)
        labels = gt < self.num_seg_cls
        gt = gt[labels].int()

        logits = output.permute(0, 2, 3, 1).contiguous().view(-1, self.num_seg_cls)
        logits = logits[labels]
        err = self.cross_entropy(logits, gt.long())

        prediction = torch.argmax(output, dim=1)
        prediction = prediction.unsqueeze(1)
        prediction = prediction.view(-1)
        prediction = prediction[labels].int()
        pixelAcc = (gt == prediction).float().mean()
        return prediction.cpu().detach().numpy(), gt.cpu().detach().numpy(), pixelAcc.cpu().detach().numpy(), err.cpu().detach().numpy()
    
    def __seg_records(self, pred, gt):
        pred, gt, pixelAcc, err = self.__seg_error(pred, gt)
        new_mat = confusion_matrix(gt, pred, self.records['labels'])
        self.records['conf_mat'] += new_mat
        self.records['pixelAccs'].append(pixelAcc)
        self.records['errs'].append(err)
        
    def __sn_error(self, pred, gt, mask):
        output = self.resize_pred(pred, gt)
        
        prediction = output.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        gt = gt.permute(0, 2, 3, 1).contiguous().view(-1, 3)

        labels = gt.max(dim=1)[0] != 255
        if mask is not None:
            gt_mask = mask.permute(0, 2, 3, 1).contiguous().view(-1, 3)
            labels = torch.logical_and(labels, gt_mask[:, 0].int() == 1)

        gt = gt[labels]
        prediction = prediction[labels]
        gt = F.normalize(gt.float(), dim=1)
        prediction = F.normalize(prediction, dim=1)

        cos_similarity = self.cosine_similiarity(gt, prediction)
        return cos_similarity.cpu().detach().numpy()
    
    def __sn_records(self, pred, gt, mask):
        cos_similarity = self.__sn_error(pred, gt, mask)
        self.records['cos_similaritys'].append(cos_similarity)
        
    def __depth_error(self, pred, gt, mask):
        output = self.resize_pred(pred, gt)
        
        if mask is not None:
            binary_mask = (gt != 255) * (mask.int() == 1)
        else:
            binary_mask = (torch.sum(gt, dim=1) > 3 * 1e-5).unsqueeze(1).cuda()
        
        depth_output_true = output.masked_select(binary_mask)
        depth_gt_true = gt.masked_select(binary_mask)
        abs_err = torch.abs(depth_output_true - depth_gt_true)
        rel_err = torch.abs(depth_output_true - depth_gt_true) / depth_gt_true
        sq_rel_err = torch.pow(depth_output_true - depth_gt_true, 2) / depth_gt_true
        abs_err = torch.sum(abs_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)
        rel_err = torch.sum(rel_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)
        sq_rel_err = torch.sum(sq_rel_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)
        # calcuate the sigma
        term1 = depth_output_true / depth_gt_true
        term2 = depth_gt_true / depth_output_true
        ratio = torch.max(torch.stack([term1, term2], dim=0), dim=0)
        # calcualte rms
        rms = torch.pow(depth_output_true - depth_gt_true, 2)
        rms_log = torch.pow(torch.log10(depth_output_true + 1e-7) - torch.log10(depth_gt_true + 1e-7), 2)

        return abs_err.cpu().detach().numpy(), rel_err.cpu().detach().numpy(), sq_rel_err.cpu().detach().numpy(), \
               ratio[0].cpu().detach().numpy(), rms.cpu().detach().numpy(), rms_log.cpu().detach().numpy()
    
    def __depth_records(self, pred, gt, mask):
        abs_err, rel_err, sq_rel_err, ratio, rms, rms_log = self.__depth_error(pred, gt, mask)
        self.records['abs_errs'].append(abs_err)
        self.records['rel_errs'].append(rel_err)
        self.records['sq_rel_errs'].append(sq_rel_err)
        self.records['ratios'].append(ratio)
        self.records['rms'].append(rms)
        self.records['rms_log'].append(rms_log)
        
    def __keypoint_edge_error(self, pred, gt):
        output = self.resize_pred(pred, gt)
        binary_mask = (gt != 255)
        keypoint_output_true = output.masked_select(binary_mask)
        keypoint_gt_true = gt.masked_select(binary_mask)
        abs_err = torch.abs(keypoint_output_true - keypoint_gt_true).mean()
        return abs_err.cpu().detach().numpy()
    
    def __keypoint_edge_records(self, pred, gt):
        err = self.__keypoint_edge_error(pred, gt)
        self.records['errs'].append(err)
    
    # Call for each batch
    def __call__(self, pred, gt, mask=None):
        self.batch_size.append(len(gt))
        
        if self.task is 'segment_semantic':
            self.__seg_records(pred, gt)
        elif self.task is 'normal':
            self.__sn_records(pred, gt, mask)
        elif self.task is 'depth_zbuffer':
            self.__depth_records(pred, gt, mask)
        elif self.task is 'keypoints2d' or self.task is 'edge_texture':
            self.__keypoint_edge_records(pred, gt)
        return
    
    # Helper function
    def round_dict(self, d):
        res = {key : round(d[key], 4) for key in d}
        return res
    
    def depth_records_modify(self):
        self.records['abs_errs'] = np.stack(self.records['abs_errs'], axis=0)
        self.records['rel_errs'] = np.stack(self.records['rel_errs'], axis=0)
        self.records['sq_rel_errs'] = np.stack(self.records['sq_rel_errs'], axis=0)
        self.records['ratios'] = np.concatenate(self.records['ratios'], axis=0)
        self.records['rms'] = np.concatenate(self.records['rms'], axis=0)
        self.records['rms_log'] = np.concatenate(self.records['rms_log'], axis=0)
        self.records['rms_log'] = self.records['rms_log'][~np.isnan(self.records['rms_log'])]
        return
    
class TaskonomyMetrics(DataMetrics):
    def __init__(self, task):
        super(TaskonomyMetrics, self).__init__(task)
        if self.task is 'segment_semantic':
            self.num_seg_cls = 17
        
        self.define_loss()
        self.define_refer()
        self.reset_records()
        
    def define_loss(self):
        super(TaskonomyMetrics, self).define_loss()
        weight = torch.from_numpy(np.load('data/utils/semseg_prior_factor.npy')).cuda().float()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, ignore_index=255)
        
    def define_refer(self):
        if self.task is 'segment_semantic':
            self.refer = {'err': 0.517}
        elif self.task is 'normal':
            self.refer = {'cosine_similarity': 0.716}
        elif self.task is 'depth_zbuffer':
            self.refer = {'abs_err': 0.021}
        elif self.task is 'keypoints2d':
            self.refer = {'err': 0.197}
        elif self.task is 'edge_texture':
            self.refer = {'err': 0.212}
        
     # Call after evaluate all data in the set
    def val_metrics(self):
        if self.task is 'segment_semantic':
            val_metrics = self.__seg_metrics()
        elif self.task is 'normal':
            val_metrics = self.__sn_metrics()
        elif self.task is 'depth_zbuffer':
            val_metrics = self.__depth_metrics()
        elif self.task is 'keypoints2d' or self.task is 'edge_texture':
            val_metrics = self.__keypoint_edge_metrics()
        self.reset_records()
        return self.round_dict(val_metrics)
    
    # Private functions
    def __seg_metrics(self):
        val_metrics = {}
        val_metrics['err'] = (np.array(self.records['errs']) * np.array(self.batch_size)).sum() / sum(self.batch_size)
        val_metrics['cmp'] = (self.refer['err'] - val_metrics['err']) / self.refer['err']
        return val_metrics
    
    def __sn_metrics(self):
        val_metrics = {}
        overall_cos = np.clip(np.concatenate(self.records['cos_similaritys']), -1, 1)
        val_metrics['cosine_similarity'] = overall_cos.mean()
        val_metrics['cmp'] = (val_metrics['cosine_similarity'] - self.refer['cosine_similarity']) / self.refer['cosine_similarity']
        return val_metrics
    
    def __depth_metrics(self):
        val_metrics = {}
        self.depth_records_modify()
        val_metrics['abs_err'] = (self.records['abs_errs'] * np.array(self.batch_size)).sum() / sum(self.batch_size)
        val_metrics['cmp'] = (self.refer['abs_err'] - val_metrics['abs_err']) / self.refer['abs_err']
        return val_metrics
    
    def __keypoint_edge_metrics(self):
        val_metrics = {}
        val_metrics['err'] = (self.records['errs'] * np.array(self.batch_size)).sum() / sum(self.batch_size)
        val_metrics['cmp'] = (self.refer['err'] - val_metrics['err']) / self.refer['err']
        return val_metrics
    
class CityScapesMetrics(DataMetrics):
    def __init__(self, task):
        super(CityScapesMetrics, self).__init__(task)
        if self.task is 'segment_semantic':
            self.num_seg_cls = 19
        
        self.define_loss()
        self.define_refer()
        self.reset_records()
        
    def define_loss(self):
        super(CityScapesMetrics, self).define_loss()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-1)
        
    def define_refer(self):
        if self.task is 'segment_semantic':
            self.refer = {'mIoU': 0.402, 'Pixel Acc': 0.747}
        elif self.task is 'depth_zbuffer':
            self.refer = {'abs_err': 0.017, 'rel_err': 0.33, 'sigma_1.25': 70.3, 'sigma_1.25^2': 86.3, 'sigma_1.25^3': 93.3}
        
     # Call after evaluate all data in the set
    def val_metrics(self):
        if self.task is 'segment_semantic':
            val_metrics = self.__seg_metrics()
        elif self.task is 'depth_zbuffer':
            val_metrics = self.__depth_metrics()
        self.reset_records()
        return self.round_dict(val_metrics)
    
    # Private functions
    def __seg_metrics(self):
        val_metrics = {}
        jaccard_perclass = []
        for i in range(self.num_seg_cls):
            if not self.records['conf_mat'][i, i] == 0:
                jaccard_perclass.append(self.records['conf_mat'][i, i] / (np.sum(self.records['conf_mat'][i, :]) + 
                                                       np.sum(self.records['conf_mat'][:, i]) -
                                                       self.records['conf_mat'][i, i]))

        val_metrics['mIoU'] = np.sum(jaccard_perclass) / len(jaccard_perclass)
        val_metrics['Pixel Acc'] = (np.array(self.records['pixelAccs']) * np.array(self.batch_size)).sum() / sum(self.batch_size)
        
        val_metrics['cmp'] = (((val_metrics['mIoU'] - self.refer['mIoU'] ) / self.refer['mIoU']) +
                       ((val_metrics['Pixel Acc'] - self.refer['Pixel Acc'] ) / self.refer['Pixel Acc'])) /2 
        return val_metrics
    
    
    def __depth_metrics(self):
        val_metrics = {}
        self.depth_records_modify()
        val_metrics['abs_err'] = (self.records['abs_errs'] * np.array(self.batch_size)).sum() / sum(self.batch_size)
        val_metrics['rel_err'] = (self.records['rel_errs'] * np.array(self.batch_size)).sum() / sum(self.batch_size)
        val_metrics['sigma_1.25'] = np.mean(np.less_equal(self.records['ratios'], 1.25)) * 100
        val_metrics['sigma_1.25^2'] = np.mean(np.less_equal(self.records['ratios'], 1.25 ** 2)) * 100
        val_metrics['sigma_1.25^3'] = np.mean(np.less_equal(self.records['ratios'], 1.25 ** 3)) * 100
        
        val_metrics['cmp'] = (((self.refer['abs_err'] - val_metrics['abs_err']) / self.refer['abs_err']) + 
                       ((self.refer['rel_err'] - val_metrics['rel_err']) / self.refer['rel_err']) + 
                       ((val_metrics['sigma_1.25'] - self.refer['sigma_1.25'] ) / self.refer['sigma_1.25']) +
                       ((val_metrics['sigma_1.25^2'] - self.refer['sigma_1.25^2'] ) / self.refer['sigma_1.25^2']) +
                       ((val_metrics['sigma_1.25^3'] - self.refer['sigma_1.25^3'] ) / self.refer['sigma_1.25^3'])) /5
        return val_metrics
    
class NYUMetrics(DataMetrics):
    def __init__(self, task, task_num=3):
        super(NYUMetrics, self).__init__(task)
        if self.task is 'segment_semantic':
            self.num_seg_cls = 40
        self.task_num = task_num
        
        self.define_loss()
        self.define_refer()
        self.reset_records()
        
    def define_loss(self):
        super(NYUMetrics, self).define_loss()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=255)
        
    def define_refer(self):
        if self.task_num == 2:
            if self.task is 'segment_semantic':
                self.refer = {'mIoU': 0.413, 'Pixel Acc': 0.691}
            elif self.task is 'normal':
                self.refer = {'Angle Mean': 15, 'Angle Median': 11.5, 'Angle 11.25': 49.2, 'Angle 22.5': 76.7, 'Angle 30': 86.8}
        elif self.task_num == 3:
            if self.task is 'segment_semantic':
                self.refer = {'mIoU': 0.275, 'Pixel Acc': 0.589}
            elif self.task is 'normal':
                self.refer = {'Angle Mean': 17.5, 'Angle Median': 14.2, 'Angle 11.25': 34.9, 'Angle 22.5': 73.3, 'Angle 30': 85.7}
            elif self.task is 'depth_zbuffer':
                self.refer = {'abs_err': 0.62, 'rel_err': 0.25, 'sigma_1.25': 57.9, 'sigma_1.25^2': 85.8, 'sigma_1.25^3': 95.7}
        
     # Call after evaluate all data in the set
    def val_metrics(self):
        if self.task is 'segment_semantic':
            val_metrics = self.__seg_metrics()
        elif self.task is 'normal':
            val_metrics = self.__sn_metrics()
        elif self.task is 'depth_zbuffer':
            val_metrics = self.__depth_metrics()
        self.reset_records()
        return self.round_dict(val_metrics)
    
    # Private functions
    def __seg_metrics(self):
        val_metrics = {}
        jaccard_perclass = []
        for i in range(self.num_seg_cls):
            if not self.records['conf_mat'][i, i] == 0:
                jaccard_perclass.append(self.records['conf_mat'][i, i] / (np.sum(self.records['conf_mat'][i, :]) + 
                                                       np.sum(self.records['conf_mat'][:, i]) -
                                                       self.records['conf_mat'][i, i]))

        val_metrics['mIoU'] = np.sum(jaccard_perclass) / len(jaccard_perclass)
        val_metrics['Pixel Acc'] = (np.array(self.records['pixelAccs']) * np.array(self.batch_size)).sum() / sum(self.batch_size)
        
        val_metrics['cmp'] = (((val_metrics['mIoU'] - self.refer['mIoU'] ) / self.refer['mIoU']) +
                       ((val_metrics['Pixel Acc'] - self.refer['Pixel Acc'] ) / self.refer['Pixel Acc'])) /2 
        return val_metrics
    
    def __sn_metrics(self):
        val_metrics = {}
        overall_cos = np.clip(np.concatenate(self.records['cos_similaritys']), -1, 1)
        angles = np.arccos(overall_cos) / np.pi * 180.0
        
        val_metrics['Angle Mean'] = np.mean(angles)
        val_metrics['Angle Median'] = np.median(angles)
        val_metrics['Angle 11.25'] = np.mean(np.less_equal(angles, 11.25)) * 100
        val_metrics['Angle 22.5'] = np.mean(np.less_equal(angles, 22.5)) * 100
        val_metrics['Angle 30'] = np.mean(np.less_equal(angles, 30.0)) * 100
        
        val_metrics['cmp'] = (((self.refer['Angle Mean'] - val_metrics['Angle Mean']) / self.refer['Angle Mean']) + 
                       ((self.refer['Angle Median'] - val_metrics['Angle Median']) / self.refer['Angle Median']) + 
                       ((val_metrics['Angle 11.25'] - self.refer['Angle 11.25'] ) / self.refer['Angle 11.25']) +
                       ((val_metrics['Angle 22.5'] - self.refer['Angle 22.5'] ) / self.refer['Angle 22.5']) +
                       ((val_metrics['Angle 30'] - self.refer['Angle 30'] ) / self.refer['Angle 30'])) /5
        return val_metrics
    
    def __depth_metrics(self):
        val_metrics = {}
        self.depth_records_modify()
        val_metrics['abs_err'] = (self.records['abs_errs'] * np.array(self.batch_size)).sum() / sum(self.batch_size)
        val_metrics['rel_err'] = (self.records['rel_errs'] * np.array(self.batch_size)).sum() / sum(self.batch_size)
        val_metrics['sigma_1.25'] = np.mean(np.less_equal(self.records['ratios'], 1.25)) * 100
        val_metrics['sigma_1.25^2'] = np.mean(np.less_equal(self.records['ratios'], 1.25 ** 2)) * 100
        val_metrics['sigma_1.25^3'] = np.mean(np.less_equal(self.records['ratios'], 1.25 ** 3)) * 100
        
        val_metrics['cmp'] = (((self.refer['abs_err'] - val_metrics['abs_err']) / self.refer['abs_err']) + 
                       ((self.refer['rel_err'] - val_metrics['rel_err']) / self.refer['rel_err']) + 
                       ((val_metrics['sigma_1.25'] - self.refer['sigma_1.25'] ) / self.refer['sigma_1.25']) +
                       ((val_metrics['sigma_1.25^2'] - self.refer['sigma_1.25^2'] ) / self.refer['sigma_1.25^2']) +
                       ((val_metrics['sigma_1.25^3'] - self.refer['sigma_1.25^3'] ) / self.refer['sigma_1.25^3'])) /5
        return val_metrics
      