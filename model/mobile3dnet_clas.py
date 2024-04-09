import torch
from torch import nn
from torch.nn import functional as F

from model.enet import InputEmbedding

class Mobile3DNet(nn.Module):
    def __init__(self, num_classes=8):
        super(Mobile3DNet, self).__init__()
        #inverted res + conv #stride 4

        self.layer0 = InputEmbedding(1, 16) # stride 2 - segcloud 7x7 in + pool
        self.bn0 = nn.BatchNorm3d(16, momentum = 0.01)
        self.swish = nn.Hardswish(inplace=True)

        self.layer1 = InvertedRes(16, 8, 16, stride = 2) # stride 4
        self.layer2 = InvertedRes(16, 72, 24)
        self.layer3 = InvertedRes(24, 88, 24)
        self.layer4 = InvertedRes(24, 96, 40)

        self.layer5 = InvertedRes(40, 240, 40, stride = 2) # stride 8
        self.layer6 = InvertedRes(40, 240, 40)
        self.layer7 = InvertedRes(40, 120, 48)
        self.layer8 = InvertedRes(48, 144, 48)
        self.layer9 = InvertedRes(48, 288, 96)

        self.layer10 = InvertedRes(96, 576, 96, stride = 2) # stride 16
        self.layer11 = InvertedRes(96, 576, 96)

        self.conv1 = nn.Conv3d(96, 576, 1, bias = False) # stride 16
        self.bn1 = nn.BatchNorm3d(576, momentum=0.01)

        self.pool = nn.AdaptiveAvgPool3d((1,1,1))

        self.fc1 = nn.Linear(576, 1000, bias=False)
        self.fc2 = nn.Linear(1000, num_classes, bias=False)

    def forward(self, x):

        x = self.layer0(x)
        x = self.bn0(x)
        x = self.swish(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)

        x = self.layer10(x)
        x = self.layer11(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.swish(x)

        x = self.pool(x).squeeze(-1).squeeze(-1).squeeze(-1) # remove tailing layers
        x = self.fc1(x)
        x = self.swish(x)
        x = self.fc2(x)
        return x



class InvertedRes(nn.Module):
    def __init__(self, inch, midch, outch, stride = 1, dilation = 1, padding = 2):
        super(InvertedRes, self).__init__()

        self.stride = stride
        self.cast = inch != outch

        self.block0 = Block(inch, midch)
        self.block1 = Block(midch, midch, stride = stride, padding = padding, dilation = dilation)
        self.block2 = Block(midch, outch)
        if self.stride > 1:
            self.pool = nn.MaxPool3d(stride)
        if self.cast:
            self.castconv = nn.Conv3d(inch, outch, 1, bias = False)

    def forward(self, x0):

        x = self.block0(x0)     # thin -> fat
        x = self.block1(x)      # fat -> fat
        x = self.block2(x)      # fat -> thin
        x0 = self.castconv(x0) if self.cast else x0
        x = x + self.pool(x0) if self.stride > 1 else x+x0 # residual

        return x



class Block(nn.Module):
    def __init__(self, inch, outch, stride = 1, dilation = 1, padding = 2):
        super(Block, self).__init__()

        self.conv0 = nn.Conv3d(inch, inch, 5, padding = padding, dilation = dilation, bias = False, stride = stride, groups = inch) #groups = inch -> unico filtro
        self.bn0 = nn.BatchNorm3d(inch, momentum=0.01)

        self.conv1 = nn.Conv3d(inch, outch, 1, bias = False)
        self.bn1 = nn.BatchNorm3d(outch, momentum=0.01)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv0(x)
        x = self.bn0(x)

        x = self.relu(x)

        x = self.conv1(x)
        x = self.bn1(x)

        x = self.relu(x)

        return x
