import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super(Block, self).__init__()
        self.expansion = 4  # Number of channels out/in
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)  
        self.relu = nn.ReLU()  
        self.downsample = downsample  

    def forward(self, x):
        identity = x  # Skip connection
        
        x = self.conv1(x)
        x = self.bn1(x)  
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)  

        if self.downsample is not None:
            identity = self.downsample(identity)  

        x += identity
        x = self.relu(x)
        return x
