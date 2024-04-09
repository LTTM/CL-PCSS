import torch
from torch import nn
from torch.nn import functional as F

class SimpleNet(nn.Module):
    def __init__(self, numclasses):
        super(SimpleNet, self).__init__()
        
        self.bn00 = nn.BatchNorm3d(1)
        self.conv00 = nn.Conv3d(1, 16, 1, bias=False)
        
        self.conv01 = nn.Conv3d(16, 16, (7,1,1), padding=(3,0,0), bias=False)
        self.conv02 = nn.Conv3d(16, 16, (1,7,1), padding=(0,3,0), bias=False)
        self.conv03 = nn.Conv3d(16, 16, (1,1,7), padding=(0,0,3), bias=False)
        self.bn01 = nn.BatchNorm3d(16)
        
        self.conv10 = nn.Conv3d(16, 32, 1, bias=False)
        self.conv11 = nn.Conv3d(32, 32, (7,1,1), padding=(3,0,0), bias=False)
        self.conv12 = nn.Conv3d(32, 32, (1,7,1), padding=(0,3,0), bias=False)
        self.conv13 = nn.Conv3d(32, 32, (1,1,7), padding=(0,0,3), bias=False)
        self.bn11 = nn.BatchNorm3d(32)
        
        self.conv20 = nn.Conv3d(32, 48, 1, bias=False)
        self.conv21 = nn.Conv3d(48, 48, (7,1,1), padding=(3,0,0), bias=False)
        self.conv22 = nn.Conv3d(48, 48, (1,7,1), padding=(0,3,0), bias=False)
        self.conv23 = nn.Conv3d(48, 48, (1,1,7), padding=(0,0,3), bias=False)
        self.bn21 = nn.BatchNorm3d(48)
        
        self.conv30 = nn.Conv3d(48, 32, 1, bias=False)
        self.conv31 = nn.Conv3d(32, 32, (7,1,1), dilation=3, padding=(9,0,0), bias=False)
        self.conv32 = nn.Conv3d(32, 32, (1,7,1), dilation=3, padding=(0,9,0), bias=False)
        self.conv33 = nn.Conv3d(32, 32, (1,1,7), dilation=3, padding=(0,0,9), bias=False)
        self.bn31 = nn.BatchNorm3d(32)
        
        #self.out1 = nn.Conv3d(16, numclasses, 3, padding=1, bias=False)
        #self.out2 = nn.Conv3d(32, numclasses, 3, padding=1, bias=False)
        #self.out3 = nn.Conv3d(48, numclasses, 3, padding=1, bias=False)
        #self.out4 = nn.Conv3d(32, numclasses, 3, padding=1, bias=False)
        self.cast = nn.Conv3d(128, 48, 1, bias=False)
        self.cbn = nn.BatchNorm3d(48)
        self.out = nn.Conv3d(48, numclasses, 3, padding=1, bias=False)
        
        self.outbn = nn.BatchNorm3d(numclasses)
        
        self.pool = nn.MaxPool3d(3, 2, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
    
        x = self.bn00(x)
        x = self.conv00(x)
        x0 = self.relu(x)
        #print(x.shape)
        
        x = self.conv01(x0)
        x = self.conv02(x)
        x = self.conv03(x)
        x = self.relu(x)
        x = self.bn01(x)+x0
        x1 = self.pool(x) # stride 2
        #print(x.shape)
        
        x0 = self.conv10(x1)
        x = self.relu(x0)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.relu(x)
        x = self.bn11(x)+x0
        x2 = self.pool(x) # stride 4
        #print(x.shape)
        
        x0 = self.conv20(x2)
        x = self.relu(x0)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.conv23(x)
        x = self.relu(x)
        x3 = self.bn21(x)+x0
        
        x0 = self.conv30(x3)
        x = self.relu(x0)
        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)
        x = self.relu(x)
        x4 = self.bn31(x)+x0
        
        #x = self.pool(self.out1(x1)) + self.out2(x2) + self.out3(x3) + self.out4(x4)
        x = torch.cat([self.pool(x1), x2, x3, x4], dim=1)
        x = self.cast(x)
        x = self.cbn(x)

        x = self.out(x)
        x = F.interpolate(x, scale_factor=4, mode='trilinear', align_corners=True)
        x = self.outbn(x)
        #print(x.shape)
        return x