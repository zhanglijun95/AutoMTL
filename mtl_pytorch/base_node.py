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
        self.taskSp = False

    def build_layer(self):
        """ pipeline for generating AutoMTL computation Node
        """
        # Step 1: Generate Basic operators depend on the user's input
        self.generate_basicOp()
        self.set_output_channels()
        #self.set_depth()

        # Step 2: Generate Task specific Operator
        self.generate_taskOp()

        #self.generate_dsOp()
        return

    def generate_basicOp(self):
        """ basic operator depends on specific layer
        """
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
                return self.compute_mtl(task, tau, hard)
            else:
                sys.exit('Only 1 task in the multi-task model. Please try stage="common".')
        else:
            sys.exit('No forward function for the given stage.')

    def compute_common(self, x):
        return self.basicOp(x)

    def compute_hard_sharing(self, x):
        return self.compute_common(x)

    def compute_specific(self, task):
        return self.taskOp[task](x);

    def compute_task_weights(self, x, task):
        if self.taskSp:
            return self.compute_common()

