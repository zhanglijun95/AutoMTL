import sys
sys.path.append('../')
import time
import numpy as np
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument('--projectroot', action='store', dest='projectroot', default='/work/lijunzhang_umass_edu/data/policymtl/', help='project directory')

parser.add_argument('--data', action='store', dest='data', default='Cityscapes', help='experiment dataset')
parser.add_argument('--batch_size', action='store', dest='bz', default=16, type=int, help='dataset batch size')
parser.add_argument('--data_dir', action='store', dest='data_dir', default='data/', help='datasets directory')

parser.add_argument('--backbone', action='store', dest='backbone', default='resnet34', help='backbone model')
parser.add_argument('--ckpt_dir', action='store', dest='ckpt_dir', default='checkpoint/Cityscapes/paper/', help='checkpoints directory')
parser.add_argument('--save_dir', action='store', dest='save_dir', default='no_pre/', help='save model directory')

parser.add_argument('--seed', action='store', dest='seed', default=10, type=int, help='random seed')
parser.add_argument('--shared', action='store', dest='shared', default=0, type=int, help='the number of bottom shared layers')
parser.add_argument('--task_loss_lambda', action='store', nargs='+', dest='task_loss_lambda', default=None, type=int, help='task loss weights')

###
parser.add_argument('--pre_train', action='store_true', help='whether to pre-train')
parser.add_argument('--fine_tune', action='store_true', help='whether to fine-tune')
parser.add_argument('--policy_lambda', action='store', dest='policy_lambda', default=0.0005, type=float, help='policy lambda')
parser.add_argument('--direct_retrain', action='store_true', help='specical case: direct retrain')

args = parser.parse_args()
print(args, flush=True)

########################### Data ##########################
print('Generate Data...', flush=True)

dataroot = os.path.join(args.projectroot, args.data_dir, args.data, '')

tasks = ['segment_semantic', 'depth_zbuffer']
task_cls_num = {'segment_semantic': 19, 'depth_zbuffer': 1}

headsDict = nn.ModuleDict()
trainDataloaderDict = {task: [] for task in tasks}
valDataloaderDict = {}
criterionDict = {}
metricDict = {}

if args.backbone == 'resnet34':
    feature_dim = 512
elif args.backbone == 'mobilenet' or args.backbone == 'mnasnet':
    feature_dim = 1280
else:
    print('Unsupported bakcbone!', flush=True)
    exit()

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
    
######################## Model #############################
print('Generate Model...', flush=True)

prototxt = '../models/deeplab_resnet34_adashare.prototxt'
mtlmodel = MTLModel(prototxt, headsDict)
mtlmodel = mtlmodel.cuda()

######################## Loss Lambda #############################
print('Generate Loss Lambda...', flush=True)

loss_lambda = {}
if args.task_loss_lambda is not None:
    for content in zip(tasks, args.task_loss_lambda):
        loss_lambda[content[0]] = content[1]
else:
    for task in tasks:
        loss_lambda[task] = 1
loss_lambda['policy'] = args.policy_lambda

######################## Trainer ###########################
print('Generate Trainer...', flush=True)
trainer = Trainer(mtlmodel, trainDataloaderDict, valDataloaderDict, criterionDict, metricDict, 
                  print_iters=100, val_iters=500, save_num=1, policy_update_iters=100) 
####!!!!!!! 50, 200, no save iters

################################### Specical: Direct Retrain ##############################
if args.direct_retrain:
    ckptpath = os.path.join(args.projectroot, args.ckpt_dir, args.save_dir, '')
    sample_path = ckptpath + str(args.shared) + '-' + str(args.seed) + '-' + str(args.policy_lambda) + '/'
    
    start_time = time.time()
    trainer.post_train(iters=30000, lr=0.001, 
                           decay_lr_freq=4000, decay_lr_rate=0.5,
                           loss_lambda=loss_lambda,
                           savePath=sample_path, reload='sample_policy_seed' + str(args.seed)+'.model')
    print("--- %s seconds ---" % (time.time() - start_time))
    exit()
##############################################################################################
    
##################### Alter Train #########################
ckptpath = os.path.join(args.projectroot, args.ckpt_dir, args.save_dir, '')

print('################# Alter Train ##############', flush=True)
if args.pre_train:
    print('Load Pretrained Weights...', flush=True)
    trainer.alter_train_with_reg(iters=500, policy_network_iters=(100,400), policy_lr=0.01, network_lr=0.0001, loss_lambda=loss_lambda, savePath=ckptpath, reload='pre_train_all_10000iter.model')
else:
    trainer.alter_train_with_reg(iters=500, policy_network_iters=(100,400), policy_lr=0.01, network_lr=0.0001, loss_lambda=loss_lambda, savePath=ckptpath)
####!!!!!!!!! iters=20000
##################### Sample Policy #####################
print('Sample Policy...', flush=True)
np.random.seed(args.seed)

policy_list = {'segment_semantic': [], 'depth_zbuffer': []}
name_list = {'segment_semantic': [], 'depth_zbuffer': []}

for name, param in mtlmodel.named_parameters():
    if 'policy' in name and not torch.eq(param, torch.tensor([0., 0., 0.]).cuda()).all():
        if 'segment_semantic' in name:
            policy_list['segment_semantic'].append(param.data.cpu().detach().numpy())
            name_list['segment_semantic'].append(name)
        elif 'depth_zbuffer' in name:
            policy_list['depth_zbuffer'].append(param.data.cpu().detach().numpy())
            name_list['depth_zbuffer'].append(name)
            
shared = args.shared
sample_policy_dict = OrderedDict()
for task in tasks:
    count = 0
    for name, policy in zip(name_list[task], policy_list[task]):
        if count < shared:
            sample_policy_dict[name] = torch.tensor([1.0,0.0,0.0]).cuda()
        else:
            distribution = softmax(policy, axis=-1)
            distribution /= sum(distribution)
            choice = np.random.choice((0,1,2), p=distribution)
            if choice == 0:
                sample_policy_dict[name] = torch.tensor([1.0,0.0,0.0]).cuda()
            elif choice == 1:
                sample_policy_dict[name] = torch.tensor([0.0,1.0,0.0]).cuda()
            elif choice == 2:
                sample_policy_dict[name] = torch.tensor([0.0,0.0,1.0]).cuda()
        count += 1
sample_path = ckptpath + str(args.shared) + '-' + str(args.seed) + '-' + str(args.policy_lambda) + '/'
if not os.path.exists(sample_path):
    os.makedirs(sample_path)
sample_state = {'state_dict': sample_policy_dict}
torch.save(sample_state, sample_path + 'sample_policy_seed' + str(args.seed)+'.model')
    
##################### Retrain / Fine-tune #####################
if args.fine_tune:
    print('################# Fine-Tune ##############', flush=True)
else:
    print('################# Retrain ##############', flush=True)
    mtlmodel = MTLModel(prototxt, headsDict)
    mtlmodel = mtlmodel.cuda()
    torch.cuda.empty_cache()
    trainer = Trainer(mtlmodel, trainDataloaderDict, valDataloaderDict, criterionDict, metricDict, 
                  print_iters=100, val_iters=500, save_num=1)
    ####!!!!!!! 50, 200, no save iters  
start_time = time.time()
trainer.post_train(iters=30000, lr=0.001, 
                       decay_lr_freq=4000, decay_lr_rate=0.5,
                       loss_lambda=loss_lambda,
                       savePath=sample_path, reload='sample_policy_seed' + str(args.seed)+'.model')
print("--- %s seconds ---" % (time.time() - start_time))