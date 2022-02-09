import torch.nn as nn
import torch
import copy
# Author: yiming huang
import google.protobuf.text_format

from framework.layer_containers import LazyLayer


class BasicNode(nn.Module):

    def __init__(self, taskList=['basic'], assumeSp=False):
        """ define the basic node/layer in MTL

        Args:
            ## protolayer: not used here since to prototxt input is given
            taskList: A list of Task that the model wants to achieve
            assumeSp:
        """
        super().__init__()
        # self.protoLayer = protoLayer no nned since no prototxt needed
        # Get from bottom: fatherNode.top = CNode.bottom and fatherNode is the deepest one with the required top
        #self.fatherNodeList = fatherNodeList  # no need, but
        self.taskList = taskList  # task list: task-specific operator()
        self.assumeSp = assumeSp  # Boolen: e.g., True for BN to be task-specific default
        self.taskOp = nn.ModuleDict()
        self.policy = nn.ParameterDict() # alpha for each task

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

class Conv2Node(BasicNode):
    def __init__(self, conv2d: nn.Conv2d, taskList=['basic'], assumpSp=False):
        """ initialize a AutoMTL-style computation Node

        Args:
            conv2d: a pytorch conv2d node
            taskList: a series of tasks that MTL want to learn
            assumpSp:
        """
        super(Conv2Node, self).__init__(taskList=taskList, assumeSp=assumpSp)
        self.taskSp = True # there are specific task for a Conv2d Node.
        self.basicOp = conv2d
        self.build_layer()

    def build_layer(self):
        super(Conv2Node, self).build_layer()

    def set_output_channels(self):
        self.outputDim = self. basicOp.out_channels


class BN2dNode(BasicNode):
    def __init__(self, bn2d: nn.BatchNorm2d, taskList=['basic'], assumpSp=False):
        super(BN2dNode, self).__init__(taskList, assumpSp)
        self.taskSp = True
        #         self.assumpSp = True
        self.layerParam = 'batch_norm_param'
        self.bn2d = bn2d
        self.build_layer()

    def build_layer(self):
        super(BN2dNode, self).build_layer()
