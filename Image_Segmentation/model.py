import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class double_conv(nn.Module):
    def __init__(self, in_plane, out_plane):
        super().__init__()
        self.layer = nn.Sequential(
                            nn.Conv2d(in_plane, out_plane, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(out_plane),
                            nn.ReLU(),
                            nn.Conv2d(out_plane, out_plane, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(out_plane),
                            nn.ReLU(),
                            )
    def forward(self, x):
        return self.layer(x)

class down(nn.Module):
    def __init__(self, in_plane, out_plane):
        super().__init__()
        self.layer = nn.Sequential(
                            nn.MaxPool2d(2, 2),
                            double_conv(in_plane, out_plane)
                            )

    def forward(self, x):
        x = self.layer(x)
        return x

class up(nn.Module):
    def __init__(self, in_plane, out_plane):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2)
        self.layer = double_conv(in_plane, out_plane)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2)) #padding left, padding right, padding top, padding bottom

        x = torch.cat([x2, x1], dim=1)
        x = self.layer(x)
        return x

"""
class Segmentation_Model(nn.Module):
    def __init__(self, n_classes=21):
        super().__init__()
        self.conv = nn.Sequential(
                            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
                            nn.BatchNorm2d(16),
                            nn.ReLU(),
        )

        self.down_1 = down(16, 32)
        self.down_2 = down(32, 64)
        self.down_3 = down(64, 128)
        self.bottle_neck = nn.Sequential(
                                double_conv(128, 256),
                                double_conv(256, 256),
                                double_conv(256, 64)
                                )
        self.up_3 = up(128, 32)
        self.up_2 = up(64, 16)
        self.up_1 = up(32, 8)

        self.out_conv = nn.Sequential(
                                nn.Conv2d(8, 128, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                                nn.Conv2d(128, n_classes, kernel_size=1, stride=1, padding=0),
                                nn.ReLU(),
                                )

    def forward(self, x):
        out_0 = self.conv(x)
        out_1 = self.down_1(out_0) #1/2
        out_2 = self.down_2(out_1) #1/4
        out_3 = self.down_3(out_2) #1/8
        out = self.bottle_neck(out_3)
        out = self.up_3(out, out_2)
        out = self.up_2(out, out_1)
        out = self.up_1(out, out_0)
        out = self.out_conv(out)

        return out
"""

class Segmentation_Model(nn.Module):
    def __init__(self, in_plane=1):
        super(Segmentation_Model, self).__init__()
        resnet = models.resnet50(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool 

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2

        self.bottleneck = nn.Sequential(
                            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                            nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0),
                            nn.BatchNorm2d(1024),
                            nn.ReLU(),
                            nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(256),
                            nn.ReLU(),
                            )

        self.upconv1 = up(512, 64)
                    
        self.upconv2 = up(128, 3)
                            
        self.upconv3 = up(6, 64)
                            

        self.out_conv = nn.Sequential(
                                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(),
                                nn.Conv2d(32, 21, kernel_size=1, stride=1, padding=0),
                                )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out_0 = self.relu(out) 
        #print('0', out_0.shape) #64x128x128
        out_1 = self.maxpool(out_0) 
        #print('0', out_1.shape) #64x64x64
        out_2 = self.layer1(out_1) 
        #print('0', out_2.shape) #256x64x64
        out_3 = self.layer2(out_2) 
        #print('0', out_3.shape) #512x32x32

        out = self.bottleneck(out_3) #512x32x32

        out = self.upconv1(out, out_2) #256 x 64 x 64
        out = self.upconv2(out, out_0)
        out = self.upconv3(out, x)
        out = self.out_conv(out)
        return out