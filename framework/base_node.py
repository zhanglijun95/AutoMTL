import torch
import torch.nn as nn

import google.protobuf.text_format
import sys
import copy 

from .layer_containers import LazyLayer

class InputNode(nn.Module):
    def __init__(self, inputDim):
        super(InputNode, self).__init__()
        self.output = None
        self.outputDim = inputDim
        self.depth = 0
    
    def set_data(self, x):
        self.output = x
        
class ComputeNode(nn.Module):
    def __init__(self, protoLayer, fatherNodeList, taskList=['basic'], assumeSp=False):
        super(ComputeNode, self).__init__()
        self.protoLayer = protoLayer
        # Get from bottom: fatherNode.top = CNode.bottom and fatherNode is the deepest one with the required top 
        self.fatherNodeList = fatherNodeList 
        self.taskList = taskList
        self.assumeSp = assumeSp          # Boolen: e.g., True for BN to be task-specific default
        
        self.layerParam = None # String: e.g., 'convolution_param' for nn.Conv2d
        self.paramMapping = {} # Dict: e.g., parameters in pytorch == parameters in prototxt for nn.Conv2d
        
        self.inputDim = None   # Int
        self.output = None
        self.outputDim = None   # Int
        self.depth = None
        
        self.basicOp = None
        self.taskSp = False            # Boolen: e.g., True for nn.Conv2d and BN
        self.taskOp = nn.ModuleDict()   # ModuleList
        self.dsOp = nn.ModuleDict() 
        self.policy = nn.ParameterDict()  # alpha for each task
        
        self.layerName = None
        self.layerTop = None
        
    def build_layer(self):
        # Function: Build CNode from the protoLayer
        # Step 1: Match the protoLayer basic properties
        self.layerName = self.protoLayer.name
        self.layerTop = self.protoLayer.top
        self.set_input_channels()

        # Step 2: Match layer attributes
        #         Create params lists, e.g. ['in_channels=3', 'out_channels=32'] for nn.Conv2d
        torchParamList = []
        if bool(self.paramMapping):
            protoParamList = getattr(self.protoLayer, self.layerParam)
            for torchAttr in self.paramMapping:
                protoAttr = self.paramMapping[torchAttr]

                # Handle Input Dimension  
                if protoAttr == 'need_input':
                    torchParamList.append(torchAttr + '=' + str(self.inputDim))
                # Handle Operator Params
                else:   
                    protoAttrCont = getattr(protoParamList, protoAttr)
                    if isinstance(protoAttrCont, google.protobuf.pyext._message.RepeatedScalarContainer):
                        protoAttrCont = protoAttrCont[0]
                    torchParamList.append(torchAttr + '=' + str(protoAttrCont))

        # Step 3: Generate basic operators 
        self.generate_basicOp(torchParamList)
        self.set_output_channels()
        self.set_depth()
        
        # Step 4: Generate task specific operatos and policy alpha
        self.generate_taskOp()
        
        # Step 5: Generate downsample part
        self.generate_dsOp()
        return
    
    def set_input_channels(self):
        # Function: Get input channels from the outputDim of fatherNodeList 
        #           inputDim: Int
        #           = fatherNodeList[0].outputDim [default for layers with 1 bottom]
        self.inputDim = self.fatherNodeList[0].outputDim
        return
    
    def set_output_channels(self):
        # Function: Set output channels according to different OpType
        #           outputDim: Int 
        #           = inputDim [default for layers that don't change dimensions]
        self.outputDim = self.inputDim
        return
    
    def set_depth(self):
        if self.taskSp and not self.assumeSp:
            self.depth = max([x.depth for x in self.fatherNodeList]) + 1
        else:
            self.depth = max([x.depth for x in self.fatherNodeList])
        return
    
    def generate_basicOp(self, torchParamList):
        # Function: Generate opCommand according to different OpType
        return
    
    def generate_taskOp(self):
        # Function: Generate task-specific operators when taskSp = True
        if len(self.taskList) > 1 and self.taskSp:
            for task in self.taskList:
                self.taskOp[task] = copy.deepcopy(self.basicOp)
                self.policy[task] = nn.Parameter(torch.tensor([0., 0., 0.])) # Default initialization
        return 
    
    def generate_dsOp(self):
        # Function: Generate downsample operators when taskSp = True and assumeSp = False
        #        Only for Conv2d now
        if len(self.taskList) > 1 and self.taskSp and not self.assumeSp:
            for task in self.taskList:
                self.dsOp[task] = nn.ModuleList()
                # First trial: Conv2d + AvgPool2d
#                 if self.basicOp.in_channels != self.basicOp.out_channels:
#                     self.dsOp[task].append(nn.Conv2d(in_channels=self.basicOp.in_channels, 
#                                           out_channels=self.basicOp.out_channels, 
#                                           kernel_size=(1,1), bias=False))
#                 if self.basicOp.stride != (1,1):
#                     self.dsOp[task].append(nn.AvgPool2d(kernel_size=self.basicOp.kernel_size,
#                                             stride=self.basicOp.stride,
#                                             padding=tuple([(k-1)//2 for k in self.basicOp.kernel_size])))
                # Second trial: Conv2d + BN
                if self.basicOp.in_channels != self.basicOp.out_channels or self.basicOp.stride != (1,1):
                    self.dsOp[task].append(nn.Conv2d(in_channels=self.basicOp.in_channels, 
                                          out_channels=self.basicOp.out_channels, 
                                          kernel_size=(1,1), 
                                          stride=self.basicOp.stride,
                                          bias=False))
                    self.dsOp[task].append(nn.BatchNorm2d(self.basicOp.out_channels))
                # For avoiding empty downsample list
                self.dsOp[task].append(LazyLayer())
        return
            
        
    def forward(self, stage, task=None, tau=5, hard=False):
        if stage == 'common':
            return self.compute_common()
        elif stage == 'hard_sharing':
            return self.compute_hard_sharing()
        elif stage == 'task_specific':
            if task is not None:
                return self.compute_task_weights(task)
            else:
                sys.exit('Please enter the specified task for stage=='+stage) 
        elif stage == 'combined' or stage == 'pre_train_all':
            if task is not None:
                return self.compute_combined(task)
            else:
                sys.exit('Please enter the specified task for stage=='+stage)
        elif stage == 'mtl':
            if len(self.taskList) > 1:
                return self.compute_mtl(task, tau, hard)
            else:
                sys.exit('Only 1 task in the multi-task model. Please try stage="common".')
        else:
            sys.exit('No forward function for the given stage.')
        
    def compute_common(self):
        # Function: Forward function when commonly used (shared with other tasks)
        #           [default for layers with 1 bottom]
        return self.basicOp(self.fatherNodeList[0].output)
        
    def compute_hard_sharing(self):
        # Function: Forward of basicOp according to different OpType
        #           For weigth pre-training by hard paramerter sharing 
        #           Logic same with compute_common()
        return self.compute_common()
    
    def copy_weight_after_pretrain(self):
        # Function: Synchronize weights of task-specific operators after the pre-train stage
        if len(self.taskList) > 1 and self.taskSp:
            for task in self.taskList:
                self.taskOp[task] = copy.deepcopy(self.basicOp)
        return
    
    def compute_specific(self, task):
        # Function: Forward function of task-specific operatos
        #           [default for layers with 1 bottom]
        return self.taskOp[task](self.fatherNodeList[0].output)
    
    def compute_task_weights(self, task):
        # Function: Forward of taskOp according to different OpType and task
        #           For weight pre-training of task-specific weights
        if self.taskSp:
            return self.compute_specific(task)
        else:
            return self.compute_common()
        
    def compute_downsample(self, task):
        # Function: Forward function of downsample operatos
        #        [Only for Conv2d now]
        x = self.fatherNodeList[0].output
        for op in self.dsOp[task]:
            x = op(x)
        return x
        
    def compute_combined(self, task):
        # Function: Forward of baiscOp, taskOp and dsOp at the same time according to different OpType and task
        #           For weight pre-training of all operators
        feature_list = []
        feature_list.append(self.compute_common())
        if self.taskSp:
            feature_list.append(self.compute_specific(task))
            if not self.assumeSp:
                feature_list.append(self.compute_downsample(task))
        return torch.mean(torch.stack(feature_list),dim=0)
            
    
    def compute_mtl(self, task, tau=5, hard=False):
        # Function: Forward of basicOp and taskOp according to different OpType
        #           For training in multitask model stage
        if self.taskSp:
            # Conv2d or BN
            if self.assumeSp:
                # BN
                return self.compute_specific(task)
            else:
                # Conv2d
                policy_task = self.policy[task]
                if hard is False:
                    # Policy-train
                    possiblity = nn.functional.gumbel_softmax(policy_task, tau=tau, hard=hard)
                    feature_common  = self.compute_common()
                    feature_specific = self.compute_specific(task)
                    feature_downsample = self.compute_downsample(task)
                    feature = feature_common*possiblity[0] + feature_specific*possiblity[1] + feature_downsample*possiblity[2]
                else:
                    # Post-train or Validation
                    branch = torch.argmax(policy_task).item()
                    if branch == 0:
                        feature = self.compute_common()
                    elif branch == 1:
                        feature = self.compute_specific(task)
                    elif branch == 2:
                        feature = self.compute_downsample(task)
                return feature
        else:
            return self.compute_common()