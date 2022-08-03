import numpy as np
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from framework.mtl_model import MTLModel
from framework.trainer import Trainer
from data.dataloader.nyuv2_dataloader import NYU_v2
from data.heads.pixel2pixel import ASPPHeadNode
from data.metrics.pixel2pixel_loss import NYUCriterions
from data.metrics.pixel2pixel_metrics import NYUMetrics

from data.dataloader.cityscapes_dataloader import CityScapes
from data.heads.pixel2pixel import ASPPHeadNode
from data.metrics.pixel2pixel_loss import CityScapesCriterions
from data.metrics.pixel2pixel_metrics import CityScapesMetrics

from data.dataloader.taskonomy_dataloader import Taskonomy
from data.heads.pixel2pixel import ASPPHeadNode
from data.metrics.pixel2pixel_loss import TaskonomyCriterions
from data.metrics.pixel2pixel_metrics import TaskonomyMetrics

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', action='store', dest='data', default='Cityscapes', help='experiment dataset')
args = parser.parse_args()
print(args, flush=True)

if args.data == 'Taskonomy':
    dataroot = '/mnt/nfs/work1/huiguan/lijunzhang/policymtl/data/Taskonomy/'

    headsDict = nn.ModuleDict()
    trainDataloaderDict = {}
    valDataloaderDict = {}
    criterionDict = {}
    metricDict = {}

    tasks = ['segment_semantic', 'normal', 'depth_zbuffer', 'keypoints2d', 'edge_texture']
    task_cls_num = {'segment_semantic': 17, 'normal': 3, 'depth_zbuffer': 1, 'keypoints2d': 1, 'edge_texture': 1}
    for task in tasks:
        headsDict[task] = ASPPHeadNode(512, task_cls_num[task])

        # For model trainer
        dataset = Taskonomy(dataroot, 'train', task, crop_h=224, crop_w=224)
        trainDataloaderDict[task] = DataLoader(dataset, batch_size=16, shuffle=True)

        dataset = Taskonomy(dataroot, 'test_small', task, crop_h=224, crop_w=224)
        valDataloaderDict[task] = DataLoader(dataset, batch_size=128, shuffle=True)

        criterionDict[task] = TaskonomyCriterions(task, dataroot)
        metricDict[task] = TaskonomyMetrics(task, dataroot)
        
    prototxt = 'models/deeplab_resnet34_adashare.prototxt'
    mtlmodel = MTLModel(prototxt, headsDict, BNsp=True)
    mtlmodel = mtlmodel.cuda()
    
    checkpoint = '/mnt/nfs/work1/huiguan/lijunzhang/policymtl/checkpoint/'
    trainer = Trainer(mtlmodel, trainDataloaderDict, valDataloaderDict, criterionDict, metricDict, 
                      print_iters=100, val_iters=50000, save_num=1, policy_update_iters=100)
    
    loss_lambda = {'segment_semantic': 1, 'normal':3, 'depth_zbuffer': 2, 'keypoints2d':7, 'edge_texture':7, 'policy':0.0005}
    trainer.alter_train_with_reg(iters=50000, policy_network_iters=(100,400), policy_lr=0.01, network_lr=0.0001, 
                                 loss_lambda=loss_lambda,
                                 savePath=checkpoint+'Taskonomy/', reload='alter_train_with_reg_0005_20000iter.model')
    
    
elif args.data == 'Cityscapes':
    dataroot = '/mnt/nfs/work1/huiguan/lijunzhang/policymtl/data/Cityscapes/'

    headsDict = nn.ModuleDict()
    trainDataloaderDict = {}
    valDataloaderDict = {}
    criterionDict = {}
    metricDict = {}

    tasks = ['segment_semantic', 'depth_zbuffer']
    task_cls_num = {'segment_semantic': 19, 'depth_zbuffer': 1}
    for task in tasks:
        headsDict[task] = ASPPHeadNode(512, task_cls_num[task])

        # For model trainer
        dataset = CityScapes(dataroot, 'train', task, crop_h=224, crop_w=224)
        trainDataloaderDict[task] = DataLoader(dataset, 16, shuffle=True)

        dataset = CityScapes(dataroot, 'test', task)
        valDataloaderDict[task] = DataLoader(dataset, 16, shuffle=True)

        criterionDict[task] = CityScapesCriterions(task)
        metricDict[task] = CityScapesMetrics(task)
        
    prototxt = 'models/deeplab_resnet34_adashare.prototxt'
    mtlmodel = MTLModel(prototxt, headsDict, BNsp=True)
    mtlmodel = mtlmodel.cuda()
    
    checkpoint = '/mnt/nfs/work1/huiguan/lijunzhang/policymtl/checkpoint/'
    trainer = Trainer(mtlmodel, trainDataloaderDict, valDataloaderDict, criterionDict, metricDict, 
                      print_iters=100, val_iters=20000, save_num=1, policy_update_iters=100)
    
    loss_lambda = {'segment_semantic': 1, 'depth_zbuffer': 1, 'policy':0.0005}
    trainer.alter_train_with_reg(iters=20000, policy_network_iters=(100,400), policy_lr=0.01, network_lr=0.0001, 
                                 loss_lambda=loss_lambda,
                                 savePath=checkpoint+'Cityscapes/', reload='pre_train_all_10000iter.model')

elif args.data == 'NYUv2':
    dataroot = '/mnt/nfs/work1/huiguan/lijunzhang/policymtl/data/NYUv2/'

    headsDict = nn.ModuleDict()
    trainDataloaderDict = {}
    valDataloaderDict = {}
    criterionDict = {}
    metricDict = {}

    tasks = ['segment_semantic', 'normal', 'depth_zbuffer']
    task_cls_num = {'segment_semantic': 40, 'normal':3, 'depth_zbuffer': 1}
    for task in tasks:
        headsDict[task] = ASPPHeadNode(512, task_cls_num[task])

        # For model trainer
        dataset = NYU_v2(dataroot, 'train', task, crop_h=321, crop_w=321)
        trainDataloaderDict[task] = DataLoader(dataset, 16, shuffle=True)

        dataset = NYU_v2(dataroot, 'test', task, crop_h=321, crop_w=321)
        valDataloaderDict[task] = DataLoader(dataset, 16, shuffle=True)

        criterionDict[task] = NYUCriterions(task)
        metricDict[task] = NYUMetrics(task)

    prototxt = 'models/deeplab_resnet34_adashare.prototxt'
    mtlmodel = MTLModel(prototxt, headsDict, BNsp=True)
    mtlmodel = mtlmodel.cuda()

    checkpoint = '/mnt/nfs/work1/huiguan/lijunzhang/policymtl/checkpoint/'
    trainer = Trainer(mtlmodel, trainDataloaderDict, valDataloaderDict, criterionDict, metricDict, 
                      print_iters=100, val_iters=20000, save_num=1, policy_update_iters=100)

    loss_lambda = {'segment_semantic': 3, 'normal':20, 'depth_zbuffer': 3, 'policy':0.003}
    trainer.alter_train_with_reg(iters=20000, policy_network_iters=(50,50), policy_lr=0.01, network_lr=0.001, 
                         loss_lambda=loss_lambda,
                         savePath=checkpoint+'NYUv2/', reload='pre_train_all_10000iter.model')
