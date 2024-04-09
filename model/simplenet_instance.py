import torch
from torch import nn
from torch.nn import functional as F


class InstDet(nn.Module):
    def __init__(self, inchs):
        super(InstDet, self).__init__()

        # block 1
        # self.conv1_block1, self.bn1_block1 = self.conv_layer(128, (3,3),(2,2)), BN(trainable=True)
        # self.conv2_block1, self.bn2_block1 = self.conv_layer(128, (3,3),(1,1)), BN(trainable=True)
        # self.conv3_block1, self.bn3_block1 = self.conv_layer(128, (3,3),(1,1)), BN(trainable=True)
        # self.conv4_block1, self.bn4_block1 = self.conv_layer(128, (3,3),(1,1)), BN(trainable=True)
        self.conv1 = SepConv(inchs, 128, 3)

        # block 2
        # self.conv1_block2, self.bn1_block2 = self.conv_layer(128, (3,3),(2,2)), BN(trainable=True)
        # self.conv2_block2, self.bn2_block2 = self.conv_layer(128, (3,3),(1,1)), BN(trainable=True)
        # self.conv3_block2, self.bn3_block2 = self.conv_layer(128, (3,3),(1,1)), BN(trainable=True)
        # self.conv4_block2, self.bn4_block2 = self.conv_layer(128, (3,3),(1,1)), BN(trainable=True)
        # self.conv5_block2, self.bn5_block2 = self.conv_layer(128, (3,3),(1,1)), BN(trainable=True)
        # self.conv6_block2, self.bn6_block2 = self.conv_layer(128, (3,3),(1,1)), BN(trainable=True)
        self.conv2 = SepConv(128, 128, 3)

        # block 3
        # self.conv1_block3, self.bn1_block3 = self.conv_layer(256, (3,3),(2,2)), BN(trainable=True)
        # self.conv2_block3, self.bn2_block3 = self.conv_layer(256, (3,3),(1,1)), BN(trainable=True)
        # self.conv3_block3, self.bn3_block3 = self.conv_layer(256, (3,3),(1,1)), BN(trainable=True)
        # self.conv4_block3, self.bn4_block3 = self.conv_layer(256, (3,3),(1,1)), BN(trainable=True)
        # self.conv5_block3, self.bn5_block3 = self.conv_layer(256, (3,3),(1,1)), BN(trainable=True)
        # self.conv6_block3, self.bn6_block3 = self.conv_layer(256, (3,3),(1,1)), BN(trainable=True)
        self.conv3 = SepConv(128, 256, 3)

        # deconvolutions
        # self.deconv_1, self.deconv_bn1 = self.deconv_layer(256, (3,3), (1,1)), BN(trainable=True)
        # self.deconv_2, self.deconv_bn2 = self.deconv_layer(256, (2,2), (2,2)), BN(trainable=True)
        # self.deconv_3, self.deconv_bn3 = self.deconv_layer(256, (4,4), (4,4)), BN(trainable=True)
        self.deconv1 = torch.nn.ConvTranspose3d(256, 256, 3)
        self.deconv2 = torch.nn.ConvTranspose3d(256, 256, 2)
        self.deconv3 = torch.nn.ConvTranspose3d(256, 256, 4)

        # probability and regression maps
        self.prob_map_conv = torch.nn.Conv3d(256, 4, 1, 1)
        self.reg_map_conv = torch.nn.Conv3d(256, 8, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # x = self.deconv1(x)
        # x = self.deconv2(x)
        # x = self.deconv3(x)

        a = self.prob_map_conv(x)
        b = self.reg_map_conv(x)

        return a, b

class SepConv(nn.Module):
    def __init__(self, inchs, outchs, kernel, dilation=1, project=True):
        super(SepConv, self).__init__()
        
        self.project = project
        if project:
            self.proj = nn.Conv3d(inchs, outchs, 1, bias=False)
            self.pbn = nn.BatchNorm3d(outchs)
        
        self.cx = nn.Conv3d(outchs, outchs, (kernel,1,1), padding=(dilation*(kernel//2),0,0), dilation=dilation, bias=False)
        self.cy = nn.Conv3d(outchs, outchs, (1,kernel,1), padding=(0,dilation*(kernel//2),0), dilation=dilation, bias=False)
        self.cz = nn.Conv3d(outchs, outchs, (1,1,kernel), padding=(0,0,dilation*(kernel//2)), dilation=dilation, bias=False)
        self.cbn = nn.BatchNorm3d(outchs)
    
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        if self.project:
            x = self.proj(x)
            x0 = self.pbn(x)
        else:
            x0 = x
        
        x = self.cx(x)
        x = self.cy(x)
        x = self.cz(x)
        x = self.cbn(x)+x0
        x = self.relu(x)
        
        return x

class SepInvRes(nn.Module):
    def __init__(self, inchs, midchs):
        super(SepInvRes, self).__init__()
        self.conv0 = nn.Conv3d(inchs, midchs, 1, bias=False)
        self.bn0 = nn.InstanceNorm3d(midchs)
        self.conv1 = SepConv(midchs, midchs, 7, project=False)
        self.bn1 = nn.InstanceNorm3d(midchs)
        self.conv2 = nn.Conv3d(midchs, inchs, 1, bias=False)
        
    def forward(self, x0):
        x = self.conv0(x0)
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)+x0
        return x

class SimpleNet(nn.Module):
    def __init__(self, num_classes=8):
        super(SimpleNet, self).__init__()
            
        self.conv0 = SepConv(1, 16, 7)
        self.pool = nn.MaxPool3d(3, 2, padding=1)
        
        self.rn1 = SepInvRes(16, 64)
        self.bn1 = nn.InstanceNorm3d(16)
        self.conv1 = SepConv(16, 8, 7)
        
        self.rn2 = SepInvRes(8, 48)
        self.bn2 = nn.InstanceNorm3d(8)
        self.conv2 = SepConv(8, 32, 7)
        
        self.rn3 = SepInvRes(32, 96)
        self.bn3 = nn.InstanceNorm3d(32)
        self.conv3 = SepConv(32, 16, 7)
        
        self.rn4 = SepInvRes(16, 64)
        self.bn4 = nn.InstanceNorm3d(16)
        self.conv4 = SepConv(16, 128, 7)

        self.inst = InstDet(128)
        
        self.out = SepConv(128, num_classes, 3)
        self.relu = nn.ReLU()#inplace=True)
        
    def forward(self, x):
    
        x = self.conv0(x)
        x = self.pool(x) #s2
        x = self.relu(x)
        
        x = self.rn1(x)
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.pool(x) #s4
        x = self.relu(x)
        
        x = self.rn2(x)
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.relu(x)

        x = self.rn3(x)
        x = self.bn3(x)
        x = self.conv3(x)
        x = self.relu(x)

        x = self.rn4(x)
        x = self.bn4(x)
        x = self.conv4(x)
        xf = self.relu(x)

        #semantic branch
        xs = self.out(xf)
        xs = F.interpolate(xs, scale_factor=4, mode='trilinear', align_corners=True)

        #offset branch
        a, b = self.inst(x)
        p = F.interpolate(a, scale_factor=4, mode='trilinear', align_corners=True)
        reg = F.interpolate(b, scale_factor=4, mode='trilinear', align_corners=True)
    
        return xs, p, reg