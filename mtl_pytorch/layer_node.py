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
                 padding: Union[size_2_t, str]= 0,
                 padding_mode: str = 'zeros',
                 dilation: Union[int, tuple] = 1,
                 taskList=['basic'],
                 assumpSp=False):
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
            assumpSp:
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
    def __init__(self, bn2d: nn.BatchNorm2d, taskList=['basic'], assumpSp=False):
        super(BN2dNode, self).__init__(taskList, assumpSp)
        nn.Conv2d
        self.taskSp = True
        #         self.assumpSp = True
        self.layerParam = 'batch_norm_param'
        self.bn2d = bn2d
        self.build_layer()

    def build_layer(self):
        super(BN2dNode, self).build_layer()
