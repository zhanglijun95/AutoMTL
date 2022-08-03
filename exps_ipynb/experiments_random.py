import numpy as np
import os
import argparse
import warnings
from pathlib import Path
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from framework.mtl_model import MTLModel
from framework.trainer import Trainer
from data.dataloader.cityscapes_dataloader import CityScapes
from data.dataloader.nyuv2_dataloader import NYU_v2
from data.dataloader.taskonomy_dataloader import Taskonomy
from data.heads.pixel2pixel import ASPPHeadNode
from data.metrics.pixel2pixel_loss import CityScapesCriterions, NYUCriterions, TaskonomyCriterions
from data.metrics.pixel2pixel_metrics import CityScapesMetrics, NYUMetrics, TaskonomyMetrics

parser = argparse.ArgumentParser()
parser.add_argument('--projectroot', action='store', dest='projectroot', default='/mnt/nfs/work1/huiguan/lijunzhang/policymtl/', help='datasets directory')

parser.add_argument('--data', action='store', dest='data', default='Cityscapes', help='experiment dataset')
parser.add_argument('--batch_size', action='store', dest='bz', default=16, type=int, help='dataset batch size')
parser.add_argument('--data_dir', action='store', dest='data_dir', default='data/', help='datasets directory')

parser.add_argument('--ckpt_dir', action='store', dest='ckpt_dir', default='checkpoint/Cityscapes-training/', help='save model directory')
parser.add_argument('--reload_ckpt', action='store', dest='reload_ckpt', default='layerwise_policy_train_20000iter.model', help='reload model parameters file')
parser.add_argument('--seed', action='store', dest='seed', default=10, type=int, help='random seed')
parser.add_argument('--shared', action='store', dest='shared', default=0, type=int, help='the number of bottom shared layers')

parser.add_argument('--val_iters', action='store', dest='val_iters', default=200, type=int, help='frequency of validation')
parser.add_argument('--print_iters', action='store', dest='print_iters', default=200, type=int, help='frequency of print')
parser.add_argument('--policy_update_iters', action='store', dest='pu_iters', default=100, type=int, help='frequency of policy update')
parser.add_argument('--save_num', action='store', dest='save_num', default=1, type=int, help='the number of saved models')
parser.add_argument('--task_iters', action='store', nargs='+', dest='task_iters', default=None, type=int, help='alternative task iterations')

parser.add_argument('--skip_random', action='store_true', help='skip generating random policy')

args = parser.parse_args()
print(args)

dataroot = os.path.join(args.projectroot, args.data_dir, args.data)
headsDict = nn.ModuleDict()
trainDataloaderDict = {}
valDataloaderDict = {}
criterionDict = {}
metricDict = {}

if args.data == 'Taskonomy':
    tasks = ['segment_semantic', 'normal', 'depth_zbuffer', 'keypoints2d', 'edge_texture']
    task_cls_num = {'segment_semantic': 17, 'normal': 3, 'depth_zbuffer': 1, 'keypoints2d': 1, 'edge_texture': 1}
    for task in tasks:
        # For Multi-task model construction
        headsDict[task] = ASPPHeadNode(512, task_cls_num[task])

        # For model trainer
        dataset = Taskonomy(dataroot, 'train', task, crop_h=224, crop_w=224)
        trainDataloaderDict[task] = DataLoader(dataset, batch_size=args.bz, shuffle=True)

        dataset = Taskonomy(dataroot, 'test_small', task,crop_h=224, crop_w=224)
        valDataloaderDict[task] = DataLoader(dataset, batch_size=args.bz, shuffle=True)

        criterionDict[task] = TaskonomyCriterions(task)
        metricDict[task] = TaskonomyMetrics(task)
        
elif args.data == 'Cityscapes':
    tasks = ['segment_semantic', 'depth_zbuffer']
    task_cls_num = {'segment_semantic': 19, 'depth_zbuffer': 1}
    for task in tasks:
        headsDict[task] = ASPPHeadNode(512, task_cls_num[task])
        
        # For model trainer
        dataset = CityScapes(dataroot, 'train', task, crop_h=224, crop_w=224)
        trainDataloaderDict[task] = DataLoader(dataset, batch_size=args.bz, shuffle=True)
        
        dataset = CityScapes(dataroot, 'test', task)
        valDataloaderDict[task] = DataLoader(dataset, batch_size=args.bz, shuffle=True)

        criterionDict[task] = CityScapesCriterions(task)
        metricDict[task] = CityScapesMetrics(task)
        
elif args.data == 'NYUv2':
    tasks = ['segment_semantic', 'normal', 'depth_zbuffer']
    task_cls_num = {'segment_semantic': 40, 'normal': 3, 'depth_zbuffer': 1}
    for task in tasks:
        headsDict[task] = ASPPHeadNode(512, task_cls_num[task])
        
        # For model trainer
        dataset = NYU_v2(dataroot, 'train', task, crop_h=321, crop_w=321)
        trainDataloaderDict[task] = DataLoader(dataset, batch_size=args.bz, shuffle=True)
        
        dataset = NYU_v2(dataroot, 'test', task, crop_h=321, crop_w=321)
        valDataloaderDict[task] = DataLoader(dataset, batch_size=args.bz, shuffle=True)

        criterionDict[task] = NYUCriterions(task)
        metricDict[task] = NYUMetrics(task)
else:
    print('Wrong dataset!')
    exit()
    
prototxt = 'models/deeplab_resnet34_adashare.prototxt'
mtlmodel = MTLModel(prototxt, headsDict, BNsp=True)
mtlmodel = mtlmodel.cuda()

######## Generate random policy #######
random_path = args.projectroot + args.ckpt_dir + 'random/' + str(args.shared) + '-' + str(args.seed) + '/'
Path(random_path).mkdir(parents=True, exist_ok=True)
if args.skip_random is False:
    torch.manual_seed(args.seed)
    state = torch.load(args.projectroot + args.ckpt_dir + args.reload_ckpt)
    mtlmodel.load_state_dict(state['state_dict'], strict=False)

    if args.data == 'Cityscapes':
        name_list = {'segment_semantic': [], 'depth_zbuffer': []}
    elif args.data == 'NYUv2':
        name_list = {'segment_semantic': [], 'normal':[], 'depth_zbuffer': []}

    for name, param in mtlmodel.named_parameters():
        if 'policy' in name and not torch.eq(param, torch.tensor([0., 0., 0.]).cuda()).all():
            if 'segment_semantic' in name:
                name_list['segment_semantic'].append(name)
            elif 'depth_zbuffer' in name:
                name_list['depth_zbuffer'].append(name)
            elif 'normal' in name:
                name_list['normal'].append(name)

    shared = args.shared
    random_policy_dict = OrderedDict()
    for task in tasks:
        count = 0
        for name in name_list[task]:
            if count < shared:
                random_policy_dict[name] = torch.tensor([1.0,0.0,0.0]).cuda()
            else:
                random_policy_dict[name] = torch.rand(3).cuda()
    random_state = {'state_dict': random_policy_dict}
    torch.save(random_state, random_path + 'random_policy_seed' + str(args.seed)+'.model')
    print('Complete Random Policy!')
    print('='*50)
#######################################

mtlmodel = MTLModel(prototxt, headsDict, BNsp=True)
mtlmodel = mtlmodel.cuda()

trainer = Trainer(mtlmodel, trainDataloaderDict, valDataloaderDict, criterionDict, metricDict, 
            print_iters=args.print_iters, val_iters=args.val_iters, policy_update_iters=args.pu_iters, save_num=args.save_num)

if args.skip_random is False:
    trainer.post_train(iters=30000, task_iters=args.task_iters, lr=0.001,
                decay_lr_freq=4000, decay_lr_rate=0.5,
                savePath=random_path, reload='random_policy_seed' + str(args.seed)+'.model')
else:
    trainer.post_train(iters=30000, task_iters=args.task_iters, lr=0.001,
                decay_lr_freq=4000, decay_lr_rate=0.5,
                savePath=random_path, reload=args.reload_ckpt)