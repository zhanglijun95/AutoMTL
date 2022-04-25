import sys
sys.path.append('/home/yiminghuang/AutoMTL/')

import torch
import torch.nn as nn
from mtl_pytorch import layer_node
from mtl_pytorch.mtl_model import mtl_model
import numpy as np

class MTL_network(mtl_model):
    def __init__(self):
        """
            A test model using mtl_pytorch API
        """
        super(MTL_network, self).__init__()
        self.convNode = layer_node.Conv2dNode(
            in_channels=1, out_channels=128, kernel_size=3,
            task_list=['segment_sementic', 'depth_zbuffer'])
        self.batchNorm = layer_node.BN2dNode(
            num_features=128
        )

    def forward(self, x, stage='common', task=None, tau=5, hard=False):
        x = self.convNode(x, stage, task, tau, hard)
        x = self.batchNorm(x, stage, task, tau, hard)
        x = x.view(-1, 256)
        x = nn.Linear(256, 1)(x)
        return x

if __name__ == '__main__':
    model = MTL_network()
    inp = np.array([[[[1, 2, 3, 4],
                      [1, 2, 3, 4],
                      [1, 2, 3, 4]]]])
    x = torch.tensor(inp, dtype=torch.float)
    for module in model.modules():
        print(module)
        print(isinstance(module, layer_node.Conv2dNode))
    print(model(x)) # forward using basic operator
    # print(model(x, stage='mtl', task='segment_sementic')) # forward with specific task operator


