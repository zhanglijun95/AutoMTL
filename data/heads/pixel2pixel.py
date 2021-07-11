import torch
import torch.nn as nn

# Heads for Pixel2Pixel
# Note: Should create for each task
#       Should be able to connect with the backbone model
class ASPPHeadNode(nn.Module):
    def __init__(self, feature_channels, out_channels):
        super(ASPPHeadNode, self).__init__()
        self.fc1 = Classification_Module(feature_channels, out_channels, rate=6)
        self.fc2 = Classification_Module(feature_channels, out_channels, rate=12)
        self.fc3 = Classification_Module(feature_channels, out_channels, rate=18)
        self.fc4 = Classification_Module(feature_channels, out_channels, rate=24)

    def forward(self, x):
        output = self.fc1(x) + self.fc2(x) + self.fc3(x) + self.fc4(x)
        return output
    
class Classification_Module(nn.Module):
    def __init__(self, inplanes, num_classes, rate=12):
        super(Classification_Module, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, 1024, kernel_size=3, stride=1, padding=rate, dilation=rate, bias=True)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.conv3 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        return x