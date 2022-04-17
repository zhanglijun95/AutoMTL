import operator
from collections import OrderedDict
from itertools import islice

from torch._jit_internal import _copy_to_script_wrapper

from mtl_pytorch.base_node import BasicNode
import torch.nn as nn
from typing import Union, Iterator
import torch
from framework.layer_containers import LazyLayer
import copy
import pydoc

size_2_t = Union[int, tuple[int, int]]


class Conv2dNode(BasicNode):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: size_2_t,
                 stride: size_2_t = 1,
                 padding: Union[size_2_t, str] = 0,
                 padding_mode: str = 'zeros',
                 dilation: Union[int, tuple] = 1,
                 bias: bool = False,
                 groups: int = 1,
                 task_list = ['basic']
                 ):
        __doc__ = r"""
        initialize a AutoMTL-style computation Node

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int, tuple or str, optional): Padding added to all four sides of
                the input. Default: 0
            padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
                ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            bias (bool, optional): If ``True``, adds a learnable bias to the
                output. Default: ``True``
            taskList: a series of tasks that MTL want to learn
        """
        super(Conv2dNode, self).__init__(taskList=task_list)
        if not isinstance(bias, bool):
            if bias is not None:
                bias = True
            else:
                bias = False
        self.taskSp = True  # there are specific task for a Conv2d Node.
        self.basicOp = nn.Conv2d(in_channels, out_channels,
                                 kernel_size, stride, padding,
                                 padding_mode=padding_mode, dilation=dilation,
                                 bias=bias, groups=groups)
        self.bias = self.basicOp.bias
        self.weight = self.basicOp.weight
        self.kernel_size = self.basicOp.kernel_size
        self.out_channels = self.basicOp.out_channels

        self.outputDim = self.basicOp.out_channels
        self.policy = nn.ParameterDict()
        self.dsOp = nn.ModuleDict()
        self.build_layer()

    def build_layer(self):
        """
            build conv2d node specific layer
        """
        super(Conv2dNode, self).build_layer()
        self.generate_dsOp()

    def generate_dsOp(self):
        """

        Returns:
            DownSample operator, needed specific for conv2d node
        """
        if len(self.taskList) > 1:
            for task in self.taskList:
                self.dsOp[task] = nn.ModuleList()
                if self.basicOp.in_channels != self.basicOp.out_channels or self.basicOp.stride != (1, 1):
                    self.dsOp[task].append(nn.Conv2d(in_channels=self.basicOp.in_channels,
                                                     out_channels=self.basicOp.out_channels,
                                                     kernel_size=(1, 1),
                                                     stride=self.basicOp.stride,
                                                     bias=False))
                    self.dsOp[task].append(nn.BatchNorm2d(self.basicOp.out_channels))
                self.dsOp[task].append(LazyLayer())
        return

    def generate_taskOp(self):
        """

        Returns:
            conv2d node specific task, extra policy array
        """
        if len(self.taskList) > 1:
            for task in self.taskList:
                self.taskOp[task] = copy.deepcopy(self.basicOp)
                self.policy[task] = nn.Parameter(torch.tensor([0., 0., 0.]))  # Default initialization
        return

    def compute_mtl(self, x, task, tau=5, hard=False):
        """

        Returns:
            output data
        """
        policy_task = self.policy[task]
        if hard is False:
            # Policy-train
            # possibility of each task
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

    def compute_combined(self, x, task):
        """

        Returns:
            Forward of basicOp, taskOp and dsOp at the same time according to different OpType and task
            For weight pre-training of all operators
        """
        feature_list = [self.compute_common(x)]
        if self.taskSp:
            feature_list.append(self.compute_specific(x, task))
            feature_list.append(self.compute_downsample(x, task))
        return torch.mean(torch.stack(feature_list), dim=0)

    def compute_downsample(self, x, task):
        """

        Returns:
            forward for downsample
        """
        for op in self.dsOp[task]:
            x = op(x)
        return x


class BN2dNode(BasicNode):  # no needed for policy
    def __init__(self,
                 num_features: int,
                 eps: float = 0.00001,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 task_list=['basic']
                 ):
        __doc__ = r"""
            Construct a Batch Norm node with search space
            Args:
                num_features: :math:`C` from an expected input of size
                :math:`(N, C, H, W)`
            eps: a value added to the denominator for numerical stability.
                Default: 1e-5
            momentum: the value used for the running_mean and running_var
                computation. Can be set to ``None`` for cumulative moving average
                (i.e. simple average). Default: 0.1
            affine: a boolean value that when set to ``True``, this module has
                learnable affine parameters. Default: ``True``
            track_running_stats: a boolean value that when set to ``True``, this
                module tracks the running mean and variance, and when set to ``False``,
                this module does not track such statistics, and initializes statistics
                buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
                When these buffers are ``None``, this module always uses batch statistics.
                in both training and eval modes. Default: ``True``
            taskList: a series of tasks that MTL want to learn
        """
        super(BN2dNode, self).__init__(task_list)
        self.basicOp = nn.BatchNorm2d(num_features,
                                      eps, momentum,
                                      affine, track_running_stats)
        self.inputDim = num_features
        self.build_layer()
        self.weight = self.basicOp.weight
        self.bias = self.basicOp.bias

    def build_layer(self):
        super(BN2dNode, self).build_layer()

    def generate_taskOp(self):
        """

        Returns:
            no need for policy
        """
        if len(self.taskList) > 1:
            for task in self.taskList:
                self.taskOp[task] = copy.deepcopy(self.basicOp)
        return

    def compute_mtl(self, x, task, tau=5, hard=False):
        """

        Returns:
            forward for mtl
        """
        return self.compute_specific(x, task)

    def compute_combined(self, x, task):
        """

        Returns: Forward of baiscOp, taskOp and dsOp at the same time according to different OpType and task
                 For weight pre-training of all operators
        """
        feature_list = []
        feature_list.append(self.compute_common(x))
        if self.taskSp:
            feature_list.append(self.compute_specific(x, task))
        return torch.mean(torch.stack(feature_list), dim=0)


class Sequential(nn.Module):

    def __init__(self, seq: nn.Sequential):
        """
            wrapper for nn.Sequential in order to apply MTL forwarding
        Args:
            seq: actual sequence of layers,
        """
        super(Sequential, self).__init__()
        self.models = seq

    def forward(self, x, stage='common', task=None, tau=5, hard=False, policy_idx=None):
        for node in self.models: # apply MTL forwarding when necessary
            if isinstance(node, Conv2dNode) or isinstance(node, BN2dNode):
                x = node(x, stage, task, tau, hard)
            else:
                x = node(x)
        return x
