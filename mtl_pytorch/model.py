import torch
import torch.nn as nn
from mtl_pytorch import layer_node
import numpy as np

class MTL_network(nn.Module):
    def __init__(self):
        super(MTL_network, self).__init__()
        self.convNode = layer_node.Conv2Node(
            in_channels=1, out_channels=128, kernel_size=3,
            taskList=['segment_sementic', ])

    def forward(self, x):
        return self.convNode(x)

if __name__ == '__main__':
    model = MTL_network()
    inp = np.array([[[[1, 2, 3, 4],
                      [1, 2, 3, 4],
                      [1, 2, 3, 4]]]])
    x = torch.tensor(inp, dtype=torch.float)
    print(model(x))