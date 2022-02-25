import torch.nn as nn
import torch
import copy
import sys
# Author: yiming huang
import google.protobuf.text_format

from framework.layer_containers import LazyLayer


class BasicNode(nn.Module):

    def __init__(self, taskList=['basic'], assumeSp=False):
        """ define the basic node/layer in MTL

        Args:
            taskList: A list of Task that the model wants to achieve
            assumeSp:
        """
        super().__init__()
        self.taskList = taskList  # task list: task-specific operator()
        self.assumeSp = assumeSp  # Boolen: e.g., True for BN to be task-specific default
        self.taskOp = nn.ModuleDict()
        self.policy = nn.ParameterDict() # alpha for each task
        self.dsOp = nn.ModuleDict() # down sample dict
        self.basicOp = None
        self.taskSp = False

    def build_layer(self):
        """ pipeline for generating AutoMTL computation Node
        """
        # Step 1: Generate Basic operators depend on the user's input
        self.generate_basicOp()
        self.set_output_channels()

        # Step 2: Generate Task specific Operator based on basic operator
        self.generate_taskOp()

        # Step 3: generate down sample part
        self.generate_dsOp()

        return

    def generate_basicOp(self):
        """ basic operator depends on specific layer
        """
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
                if self.basicOp.in_channels != self.basicOp.out_channels or self.basicOp.stride != (1, 1):
                    self.dsOp[task].append(nn.Conv2d(in_channels=self.basicOp.in_channels,
                                                     out_channels=self.basicOp.out_channels,
                                                     kernel_size=(1, 1),
                                                     stride=self.basicOp.stride,
                                                     bias=False))
                    self.dsOp[task].append(nn.BatchNorm2d(self.basicOp.out_channels))
                # For avoiding empty downsample list
                self.dsOp[task].append(LazyLayer())
        return

    def set_output_channels(self):
        """ by default, output dimension should match up with the input dimension
        """
        self.outputDim = self.inputDim
        return

    def generate_taskOp(self):
        if len(self.taskList) > 1 and self.taskSp:
            for task in self.taskList:
                self.taskOp[task] = copy.deepcopy(self.basicOp)
                self.policy[task] = nn.Parameter(torch.tensor([0., 0., 0.]))  # Default initialization
        return

    def generate_dsOp(self):
        if len(self.taskList) > 1 and self.taskSp and not self.assumeSp:
            for task in self.taskList:
                self.dsOp[task] = nn.ModuleList()
                # Second trial: Conv2d + BN
                if self.basicOp.in_channels != self.basicOp.out_channels or self.basicOp.stride != (1, 1):
                    self.dsOp[task].append(nn.Conv2d(in_channels=self.basicOp.in_channels,
                                                     out_channels=self.basicOp.out_channels,
                                                     kernel_size=(1, 1),
                                                     stride=self.basicOp.stride,
                                                     bias=False))
                    self.dsOp[task].append(nn.BatchNorm2d(self.basicOp.out_channels))
                # For avoiding empty downsample list
                self.dsOp[task].append(LazyLayer())
        return

    def forward(self, x, stage='common', task=None, tau=5, hard=False): # for every input
        """

        Args:
            x: input data
            stage: the stage for training
            task: the task list
            tau:
            hard: whether to share the common parameter

        Returns:

        """
        if stage == 'common':
            return self.compute_common(x)
        elif stage == 'hard_sharing':
            return self.compute_hard_sharing(x)
        elif stage == 'task_specific':
            if task is not None:
                return self.compute_task_weights(x, task)
            else:
                sys.exit('Please enter the specified task for stage==' + stage)
        elif stage == 'combined' or stage == 'pre_train_all':
            if task is not None:
                return self.compute_combined(task)
            else:
                sys.exit('Please enter the specified task for stage==' + stage)
        elif stage == 'mtl':
            if len(self.taskList) > 1:
                return self.compute_mtl(x, task, tau, hard)
            else:
                sys.exit('Only 1 task in the multi-task model. Please try stage="common".')
        else:
            sys.exit('No forward function for the given stage.')

    def compute_common(self, x):
        """
        Args:
            x: input data
        Returns: compute the output of basic operator

        """
        return self.basicOp(x)

    def compute_hard_sharing(self, x):
        """
        Args:
            x: input data
        Returns: compute the output for hard parameter sharing training
        """
        return self.compute_common(x)

    def copy_weight_after_pretrain(self):
        """ copy the shared operator to
        other task-specific operator for initialization
        """
        if len(self.taskList) > 1 and self.taskSp:
            for task in self.taskList:
                self.taskOp[task] = copy.deepcopy(self.baiscOp)
        return

    def compute_specific(self, x, task):
        return self.taskOp[task](x)

    def compute_downsample(self, x, task):
        for op in self.dsOp[task]:
            x = op(x)
        return x

    def compute_task_weights(self, x, task):
        """
        Args:
            x: input data
            task: the task we want to get for result
        Returns: the weights for task we want to compute
        """
        if self.taskSp:
            return self.compute_specific(x, task)
        else:
            return self.compute_common(x)

    def compute_mtl(self, x, task, tau=5, hard=False):
        # Function: Forward of basicOp and taskOp according to different OpType
        #           For training in multitask model stage
        if self.taskSp:
            # Conv2d or BN
            if self.assumeSp:
                # BN
                return self.compute_specific(x, task)
            else:
                # Conv2d
                policy_task = self.policy[task]
                if hard is False:
                    # Policy-train
                    possiblity = nn.functional.gumbel_softmax(policy_task, tau=tau, hard=hard)
                    feature_common = self.compute_common(x)
                    feature_specific = self.compute_specific(x, task)
                    feature_downsample = self.compute_downsample(x, task)
                    feature = feature_common * possiblity[0] + feature_specific * possiblity[1] + feature_downsample * \
                              possiblity[2]
                else:
                    # Post-train or Validation
                    branch = torch.argmax(policy_task).item()
                    if branch == 0:
                        feature = self.compute_common(x)
                    elif branch == 1:
                        feature = self.compute_specific(x, task)
                    elif branch == 2:
                        feature = self.compute_downsample(x, task)
                return feature
        else:
            return self.compute_common(x)

