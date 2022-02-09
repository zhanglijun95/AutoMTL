from pytorch.node_constructor import Conv2Node, BasicNode, BN2dNode
import torch.nn as nn

class MTL_network(nn.Module):
    def __init__(self):
        super(MTL_network, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            Conv2Node(),
            BN2dNode(),
            nn.Conv2d,
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits


if __name__ == '__main__':
    model = MTL_network().to()
    print(model)