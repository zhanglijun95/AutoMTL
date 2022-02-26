from mtl_pytorch.base_node import BasicNode
import torch.nn as nn
from typing import Union

size_2_t = Union[int, tuple[int, int]]

class Conv2Node(BasicNode):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: size_2_t,
                 stride: size_2_t = 1,
                 padding: Union[size_2_t, str] = 0,
                 padding_mode: str = 'zeros',
                 dilation: Union[int, tuple] = 1,
                 taskList = ['basic'],
                 assumpSp = False):
        """
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
            assumpSp: TODO: what is this for ?
        """
        super(Conv2Node, self).__init__(taskList=taskList, assumeSp=assumpSp)
        self.taskSp = True  # there are specific task for a Conv2d Node.
        self.basicOp = nn.Conv2d(in_channels, out_channels,
                                 kernel_size, stride, padding,
                                 padding_mode=padding_mode, dilation=dilation)
        self.build_layer()

    def build_layer(self):
        super(Conv2Node, self).build_layer()

    def set_output_channels(self):
        self.outputDim = self.basicOp.out_channels


class BN2dNode(BasicNode):
    def __init__(self,
                 num_features: int,
                 eps: float = 0.00001,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool =True ,
                 taskList = ['basic'],
                 assumpSp = False):
        """
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
            assumpSp:
        """
        super(BN2dNode, self).__init__(taskList, assumpSp)
        self.taskSp = True
        self.basicOp = nn.BatchNorm2d(num_features,
                                      eps, momentum,
                                      affine, track_running_stats)
        self.inputDim = num_features
        self.build_layer()

    def build_layer(self):
        super(BN2dNode, self).build_layer()
