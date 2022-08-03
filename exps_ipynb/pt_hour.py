import sys
sys.path.append('../')
import argparse
import time
import numpy as np
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
from scipy.special import softmax
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from framework.mtl_model import MTLModel
from framework.trainer import Trainer
from data.heads.pixel2pixel import ASPPHeadNode

from data.dataloader.cityscapes_dataloader import CityScapes
from data.metrics.pixel2pixel_loss import CityScapesCriterions
from data.metrics.pixel2pixel_metrics import CityScapesMetrics

from data.dataloader.nyuv2_dataloader import NYU_v2
from data.metrics.pixel2pixel_loss import NYUCriterions
from data.metrics.pixel2pixel_metrics import NYUMetrics

from data.dataloader.taskonomy_dataloader import Taskonomy
from data.metrics.pixel2pixel_loss import TaskonomyCriterions
from data.metrics.pixel2pixel_metrics import TaskonomyMetrics

parser = argparse.ArgumentParser()
parser.add_argument('--projectroot', action='store', dest='projectroot', default='/work/lijunzhang_umass_edu/data/policymtl/', help='project directory')

parser.add_argument('--data', action='store', dest='data', default='CityScapes', help='experiment dataset')
parser.add_argument('--batch_size', action='store', dest='bz', default=16, type=int, help='dataset batch size')
parser.add_argument('--data_dir', action='store', dest='data_dir', default='data/', help='datasets directory')

parser.add_argument('--backbone', action='store', dest='backbone', default='resnet34', help='backbone model')
args = parser.parse_args()
print(args, flush=True)

########################### Data ##########################
print('Generate Data...', flush=True)

dataroot = os.path.join(args.projectroot, args.data_dir, args.data, '')
headsDict = nn.ModuleDict()
valDataloaderDict = {}
criterionDict = {}
metricDict = {}

feature_dim = 512
if args.data == 'CityScapes':
    tasks = ['segment_semantic', 'depth_zbuffer']
    task_cls_num = {'segment_semantic': 19, 'depth_zbuffer': 1}
    
    trainDataloaderDict = {task: [] for task in tasks}
    for task in tasks:
        headsDict[task] = ASPPHeadNode(feature_dim, task_cls_num[task])

        # For model trainer
        dataset = CityScapes(dataroot, 'train', task, crop_h=224, crop_w=224)
        trainDataloaderDict[task].append(DataLoader(dataset, 16, shuffle=True))
        dataset1 = CityScapes(dataroot, 'train1', task, crop_h=224, crop_w=224)
        trainDataloaderDict[task].append(DataLoader(dataset1, 16, shuffle=True)) # for network param training
        dataset2 = CityScapes(dataroot, 'train2', task, crop_h=224, crop_w=224)
        trainDataloaderDict[task].append(DataLoader(dataset2, 16, shuffle=True)) # for policy param training

        dataset = CityScapes(dataroot, 'test', task)
        valDataloaderDict[task] = DataLoader(dataset, 8, shuffle=True)

        criterionDict[task] = CityScapesCriterions(task)
        metricDict[task] = CityScapesMetrics(task)
        
elif args.data == 'Taskonomy':
    tasks = ['segment_semantic', 'normal', 'depth_zbuffer', 'keypoints2d', 'edge_texture']
    task_cls_num = {'segment_semantic': 17, 'normal': 3, 'depth_zbuffer': 1, 'keypoints2d': 1, 'edge_texture': 1}
    
    trainDataloaderDict = {task: [] for task in tasks}
    for task in tasks:
        headsDict[task] = ASPPHeadNode(512, task_cls_num[task])

        # For model trainer
        dataset = Taskonomy(dataroot, 'train', task, crop_h=224, crop_w=224)
        trainDataloaderDict[task].append(DataLoader(dataset, batch_size=16, shuffle=True))
        dataset1 = Taskonomy(dataroot, 'train1', task, crop_h=224, crop_w=224)
        trainDataloaderDict[task].append(DataLoader(dataset1, 16, shuffle=True)) # for network param training
        dataset2 = Taskonomy(dataroot, 'train2', task, crop_h=224, crop_w=224) 
        trainDataloaderDict[task].append(DataLoader(dataset2, 16, shuffle=True)) # for policy param training

        dataset = Taskonomy(dataroot, 'test_small', task, crop_h=224, crop_w=224)
        valDataloaderDict[task] = DataLoader(dataset, batch_size=8, shuffle=True)

        criterionDict[task] = TaskonomyCriterions(task, dataroot)
        metricDict[task] = TaskonomyMetrics(task, dataroot)
    
elif args.data == 'NYUv2':
    tasks = ['segment_semantic', 'normal', 'depth_zbuffer']
    task_cls_num = {'segment_semantic': 40, 'normal': 3, 'depth_zbuffer': 1}
    
    trainDataloaderDict = {task: [] for task in tasks}
    for task in tasks:
        headsDict[task] = ASPPHeadNode(512, task_cls_num[task])

        # For model trainer
        dataset = NYU_v2(dataroot, 'train', task, crop_h=321, crop_w=321)
        trainDataloaderDict[task].append(DataLoader(dataset, 16, shuffle=True))
        dataset1 = NYU_v2(dataroot, 'train1', task, crop_h=321, crop_w=321)
        trainDataloaderDict[task].append(DataLoader(dataset1, 16, shuffle=True)) # for network param training
        dataset2 = NYU_v2(dataroot, 'train2', task, crop_h=321, crop_w=321) 
        trainDataloaderDict[task].append(DataLoader(dataset2, 16, shuffle=True)) # for policy param training

        dataset = NYU_v2(dataroot, 'test', task, crop_h=321, crop_w=321)
        valDataloaderDict[task] = DataLoader(dataset, 8, shuffle=True)

        criterionDict[task] = NYUCriterions(task)
        metricDict[task] = NYUMetrics(task)
    
######################## Model #############################
print('Generate Model...', flush=True)
    
prototxt = '../models/deeplab_resnet34_adashare.prototxt'
mtlmodel = MTLModel(prototxt, headsDict)
mtlmodel = mtlmodel.cuda()

######################## Loss Lambda #############################
print('Generate Loss Lambda...', flush=True)

loss_lambda = {task: 1 for task in tasks}
loss_lambda['policy'] = 0.0001

######################## Trainer ###########################
print('Generate Trainer...', flush=True)
trainer = Trainer(mtlmodel, trainDataloaderDict, valDataloaderDict, criterionDict, metricDict, 
                  print_iters=100, val_iters=500, save_num=1, policy_update_iters=100) 

##################### Retrain / Fine-tune #####################
print('################# Retrain ##############', flush=True)
start_time = time.time()
trainer.post_train(iters=1000, lr=0.001, 
                       decay_lr_freq=4000, decay_lr_rate=0.5,
                       loss_lambda=loss_lambda)
print("--- %s seconds ---" % (time.time() - start_time))