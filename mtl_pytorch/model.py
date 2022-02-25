import torch
import torch.nn as nn
from mtl_pytorch import layer_node
import numpy as np

class MTL_network(nn.Module):
    def __init__(self):
        super(MTL_network, self).__init__()
        self.convNode = layer_node.Conv2Node(
            in_channels=1, out_channels=128, kernel_size=3,
            taskList=['segment_sementic', 'depth_zbuffer'])

    def forward(self, x, stage='common', task=None, tau=5, hard=False):
        x = self.convNode(x, stage, task, tau, hard)
        x = x.view(-1, 256)
        x = nn.Linear(256, 1)(x)
        return x

if __name__ == '__main__':
    model = MTL_network()
    inp = np.array([[[[1, 2, 3, 4],
                      [1, 2, 3, 4],
                      [1, 2, 3, 4]]]])
    x = torch.tensor(inp, dtype=torch.float)

    print(model(x)) # forward using basic operator
    print(model(x, stage='mtl', task='segment_sementic')) # forward with specific task operator


