import time
import numpy as np
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# import matplotlib.pyplot as plt
import sys
sys.path.append('/home/yiminghuang/AutoMTL')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from layer_node import Conv2dNode, BN2dNode

from trainer import Trainer
from data.heads.pixel2pixel import ASPPHeadNode

from data.dataloader.cityscapes_dataloader import CityScapes
from data.metrics.pixel2pixel_loss import CityScapesCriterions
from data.metrics.pixel2pixel_metrics import CityScapesMetrics

from mobilenetv2 import mobilenet_v2

from mtl_model import mtl_model

from data.dataloader.nyuv2_dataloader import NYU_v2
from data.metrics.pixel2pixel_loss import NYUCriterions
from data.metrics.pixel2pixel_metrics import NYUMetrics

from data.dataloader.taskonomy_dataloader import Taskonomy
from data.metrics.pixel2pixel_loss import TaskonomyCriterions
from data.metrics.pixel2pixel_metrics import TaskonomyMetrics

print("start training! ")
dataroot = 'datasets/Cityscapes/'

headsDict = nn.ModuleDict()
trainDataloaderDict = {}
valDataloaderDict = {}
criterionDict = {}
metricDict = {}

tasks = ['segment_semantic', 'depth_zbuffer']
task_cls_num = {'segment_semantic': 19, 'depth_zbuffer': 1}
for task in tasks:
    headsDict[task] = ASPPHeadNode(1280, task_cls_num[task])

    # For model trainer
    dataset = CityScapes(dataroot, 'train', task, crop_h=224, crop_w=224)
    trainDataloaderDict[task] = DataLoader(dataset, 16, shuffle=True)

    dataset = CityScapes(dataroot, 'test', task)
    valDataloaderDict[task] = DataLoader(dataset, 16, shuffle=True)

    criterionDict[task] = CityScapesCriterions(task)
    metricDict[task] = CityScapesMetrics(task)

mtlmodel = mobilenet_v2(False, heads_dict=headsDict)
if torch.cuda.is_available():
    mtlmodel.cuda()

checkpoint = 'checkpoint/'
trainer = Trainer(mtlmodel, trainDataloaderDict, valDataloaderDict, criterionDict, metricDict, 
                  print_iters=100, val_iters=500, save_num=1, policy_update_iters=100)

print("start pre train")
# iters = 10000
trainer.pre_train(iters=1, lr=0.0001, savePath=checkpoint+'Cityscapes/')

loss_lambda = {'segment_semantic': 1, 'depth_zbuffer': 1, 'policy':0.0005}
print("start alter train")
# iters = 20000
trainer.alter_train_with_reg(iters=1, policy_network_iters=(100,400), policy_lr=0.01, network_lr=0.0001,
                             loss_lambda=loss_lambda,
                             savePath=checkpoint+'Cityscapes/')

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

from collections import OrderedDict
from scipy.special import softmax

sample_policy_dict = OrderedDict()
for task in tasks:
    for name, policy in zip(name_list[task], policy_list[task]):
        # distribution = softmax(policy, axis=-1
        distribution = softmax(policy, axis=-1)
        distribution /= sum(distribution)

        choice = np.random.choice((0,1,2), p=distribution)
        if choice == 0:
            sample_policy_dict[name] = torch.tensor([1.0,0.0,0.0]).cuda()
        elif choice == 1:
            sample_policy_dict[name] = torch.tensor([0.0,1.0,0.0]).cuda()
        elif choice == 2:
            sample_policy_dict[name] = torch.tensor([0.0,0.0,1.0]).cuda()

sample_path = 'checkpoint/CityScapes/'
# app.run(debug=True, use_reloader=False)
sample_state = {'state_dict': sample_policy_dict}
torch.save(sample_state, sample_path + 'sample_policy.model')

while not os.path.exists(sample_path + 'sample_policy.model'):
    print('waiting')
    time.sleep(1)

loss_lambda = {'segment_semantic': 1, 'depth_zbuffer': 1}
# iters = 20000
print('start post_train')
trainer.post_train(iters=10, lr=0.001,
                   decay_lr_freq=4000, decay_lr_rate=0.5,
                   loss_lambda=loss_lambda,
                   savePath=checkpoint+'CityScapes/',
                   reload='sample_policy.model')

torch.save(mtlmodel.state_dict(), 'CityScapes.model')