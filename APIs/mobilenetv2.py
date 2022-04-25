# A Multi-task-learning model
# based on https://github.com/tonylins/pytorch-mobilenet-v2
import warnings

import torch.nn as nn
import math
import torch
from mtl_model import mtl_model
from layer_node import Sequential, Conv2dNode, BN2dNode


def conv_bn(inp, oup, stride, task_list):
    return Sequential(
        nn.Sequential(
            Conv2dNode(inp, oup, 3, stride, 1, bias=False,
                       task_list=task_list),
            BN2dNode(oup, task_list=task_list),
            nn.ReLU6(inplace=True)
        )
    )


def conv_1x1_bn(inp, oup, task_list):
    return Sequential(
        nn.Sequential(
            Conv2dNode(inp, oup, 1, 1, 0, bias=False,
                       task_list=task_list),
            BN2dNode(oup, task_list=task_list),
            nn.ReLU6(inplace=True)
        )
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, task_list=[]):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = Sequential(
                nn.Sequential(
                    # dw
                    Conv2dNode(hidden_dim, hidden_dim, 3, stride, 1,
                               groups=hidden_dim, bias=False, task_list=task_list),
                    BN2dNode(hidden_dim, task_list=task_list),
                    nn.ReLU6(inplace=True),
                    # pw-linear
                    Conv2dNode(hidden_dim, oup, 1, 1, 0, bias=False, task_list=task_list),
                    BN2dNode(oup, task_list=task_list),
                )
            )
        else:
            self.conv = Sequential(
                nn.Sequential(
                    # pw
                    Conv2dNode(inp, hidden_dim, 1, 1, 0, bias=False, task_list=task_list),
                    BN2dNode(hidden_dim, task_list=task_list),
                    nn.ReLU6(inplace=True),
                    # dw
                    Conv2dNode(hidden_dim, hidden_dim, 3, stride, 1,
                               groups=hidden_dim, bias=False, task_list=task_list),
                    BN2dNode(hidden_dim, task_list=task_list),
                    nn.ReLU6(inplace=True),
                    # pw-linear
                    Conv2dNode(hidden_dim, oup, 1, 1, 0, bias=False, task_list=task_list),
                    BN2dNode(oup, task_list=task_list),
                )
            )

    # add-on
    def forward(self, x, stage='common', task=None, tau=5, hard=False, policy_idx=None):
        if self.use_res_connect:
            return x + self.conv(x, stage, task, tau, hard, policy_idx)
        else:
            return self.conv(x, stage, task, tau, hard, policy_idx)


# inherit in order to use trainer provided
class MobileNetV2(mtl_model):
    def __init__(self, n_class=1000, input_size=224, width_mult=1., heads_dict={}):
        super(MobileNetV2, self).__init__()

        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        self.task_list = heads_dict.keys()
        self.heads_dict = heads_dict

        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        self.c = Conv2dNode

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, task_list=self.task_list)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t, task_list=self.task_list))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t, task_list=self.task_list))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel,
                                         task_list=self.task_list))
        # make it nn.Sequential
        self.features = Sequential(nn.Sequential(*self.features))

        # building classifier (leave classification work to heads_dict)
        # self.classifier = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()
        # NEEDED: simply compute the depth for every node
        self.compute_depth()

    def forward(self, x, stage='common', task=None, tau=5, hard=False, policy_idx=None):
        # Step 1: get feature from backbone model
        feature = self.features(x, stage, task, tau, hard, policy_idx)

        # Step 2: Add heads for each task
        if task != None:
            output = self.heads_dict[task](feature)
            return output
        else:
            warnings.warn('No task specified, return feature directly')
            return feature

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2dNode):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BN2dNode):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenet_v2(pretrained=True, heads_dict={}):
    model = MobileNetV2(width_mult=1, heads_dict=heads_dict)
    if pretrained:
        try:
            from torch.hub import load_state_dict_from_url
        except ImportError:
            from torch.utils.model_zoo import load_url as load_state_dict_from_url
        state_dict = load_state_dict_from_url(
            'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1', progress=True)
        model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    net = mobilenet_v2(True)
