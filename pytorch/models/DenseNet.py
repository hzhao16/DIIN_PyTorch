import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import collections

class Dense_net_block(nn.Module):
    def __init__(self, outChannels, growth_rate, kernel_size):
        super(Dense_net_block, self).__init__()
        self.conv = nn.Conv2d(outChannels, growth_rate, kernel_size=kernel_size, bias=False, padding=1)

    def forward(self, x):
        ft = F.relu(self.conv(x))
        out = torch.cat((x, ft), dim=1)
        return out

class Dense_net_transition(nn.Module):
    def __init__(self, nChannels, outChannels):
        super(Dense_net_transition, self).__init__()
        self.conv = nn.Conv2d(nChannels, outChannels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(x)
        out = F.max_pool2d(out, (2,2), (2,2), padding=0)
        return out

class DenseNet(nn.Module):
    def __init__(self, nChannels, growthRate, reduction, nDenseBlocks, kernel_size):
        super(DenseNet, self).__init__()
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, kernel_size)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Dense_net_transition(nChannels, nOutChannels)
        nChannels = nOutChannels
        
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, kernel_size)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Dense_net_transition(nChannels, nOutChannels)
        nChannels = nOutChannels
       
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, kernel_size)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans3 = Dense_net_transition(nChannels, nOutChannels)

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, kernel_size):
        layers = []
        for i in range(int(nDenseBlocks)):
            layers.append(Dense_net_block(nChannels, growthRate, kernel_size))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.trans1(self.dense1(x))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        return out
