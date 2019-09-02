import torch
import torch.nn as nn
import torch.nn.functional as F
from unet_parts import *

class Unet_Generator(nn.Module):
    def __init__(self, ):
        super(Unet_Generator, self).__init__()
        self.inc = inconv(6, 16)
        self.down1 = down(16, 32)
        self.down2 = down(32, 64)
        self.down3 = down(64, 128)
        self.down4 = down(128, 128)
        self.bottleneck_1 = double_conv(128, 128)
        self.bottleneck_2 = double_conv(128, 128)
        self.up1 = up(256, 128)
        self.up2 = up(192, 64)
        self.up3 = up(96, 32)
        self.up4 = up(48, 16)
        self.outc = nn.Sequential(nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0), nn.CELU())

        self.global_conv_1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.CELU()) #nn.InstanceNorm2d(128),
        self.global_conv_2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.CELU())
        self.global_conv_3 = nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.CELU()) #TODO upsampling trial
        self.global_conv_4 = nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.CELU())
        self.global_linear = nn.Linear(128*8**2, 128)


    def forward(self, x):
        #TODO serperatable normalize
        #input = torch.zeros(x.size(0), x.size(1)*2, x.size(2), x.size(3))
        normalized_input = torch.zeros_like(x)
        for i in range(x.size(0)):#batch size
            for j in range(x.size(1)):#channel depth: 3
                channel_mean = x[i][j].mean()
                cahnnel_mean_std = x[i][j].std()
                normalized_input[i][j] = (x[i][j] - channel_mean) / cahnnel_mean_std
        input = torch.cat([x, normalized_input], dim=1)
        x1 = self.inc(input)
        x2 = self.down1(x1)#1/2
        x3 = self.down2(x2)#1/4
        x4 = self.down3(x3)#1/8
        x5 = self.down4(x4)#1/16

        #global_feature = self.global_conv_1(x5)#1/32
        #global_feature = self.global_conv_2(global_feature) #1/64 128x8x8
        #global_feature = self.global_conv_3(global_feature)
        #global_feature = self.global_conv_4(global_feature) #1/16
        #TODO dynamic linear F.linear(global_feature.shape....) 이러면 학습이 되나 ? 
        #global_feature = global_feature.view(x.size(0), -1)
        #global_feature = self.global_linear(global_feature).view(x.size(0), -1, 1, 1)
        #global_feature = global_feature.expand_as(x5)

        x6 = self.bottleneck_1(x5) #1/16
        #x7 = self.bottleneck_2(torch.cat((x6, global_feature), dim=1))
        x7 = self.bottleneck_2(x6)
        out = self.up1(x7, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        out = self.outc(out)
        return out+x #res

class Triple_Unet_Generator(nn.Module):
    def __init__(self, ):
        super(Triple_Unet_Generator, self).__init__()
        self.inc = triple_inconv(6, 16)
        self.down1 = triple_down(16, 32)
        self.down2 = triple_down(32, 64)
        self.down3 = triple_down(64, 128)
        self.down4 = triple_down(128, 128)
        self.bottleneck_1 = triple_conv(128, 128)
        self.bottleneck_2 = triple_conv(128, 128)
        self.up1 = triple_up(256, 128)
        self.up2 = triple_up(192, 64)
        self.up3 = triple_up(96, 32)
        self.up4 = triple_up(48, 16)
        self.outc = nn.Sequential(nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0), nn.CELU())

        self.global_conv_1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.CELU()) #nn.InstanceNorm2d(128),
        self.global_conv_2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.CELU())
        self.global_conv_3 = nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.CELU()) #TODO upsampling trial
        self.global_conv_4 = nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.CELU())
        self.global_linear = nn.Linear(128*8**2, 128)


    def forward(self, x):
        #TODO serperatable normalize
        #input = torch.zeros(x.size(0), x.size(1)*2, x.size(2), x.size(3))
        normalized_input = torch.zeros_like(x)
        for i in range(x.size(0)):#batch size
            for j in range(x.size(1)):#channel depth: 3
                channel_mean = x[i][j].mean()
                cahnnel_mean_std = x[i][j].std()
                normalized_input[i][j] = (x[i][j] - channel_mean) / cahnnel_mean_std
        input = torch.cat([x, normalized_input], dim=1)
        x1 = self.inc(input)
        x2 = self.down1(x1)#1/2
        x3 = self.down2(x2)#1/4
        x4 = self.down3(x3)#1/8
        x5 = self.down4(x4)#1/16

        #global_feature = self.global_conv_1(x5)#1/32
        #global_feature = self.global_conv_2(global_feature) #1/64 128x8x8
        #global_feature = self.global_conv_3(global_feature)
        #global_feature = self.global_conv_4(global_feature) #1/16
        #TODO dynamic linear F.linear(global_feature.shape....) 이러면 학습이 되나 ? 
        #global_feature = global_feature.view(x.size(0), -1)
        #global_feature = self.global_linear(global_feature).view(x.size(0), -1, 1, 1)
        #global_feature = global_feature.expand_as(x5)

        x6 = self.bottleneck_1(x5) #1/16
        #x7 = self.bottleneck_2(torch.cat((x6, global_feature), dim=1))
        x7 = self.bottleneck_2(x6)
        out = self.up1(x7, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        out = self.outc(out)
        return out+x #res

class Low_Unet_Generator(nn.Module):
    def __init__(self, ):
        super(Low_Unet_Generator, self).__init__()
        self.inc = low_inconv(6, 16)
        self.down1 = low_down(16, 32)
        self.down2 = low_down(32, 64)
        self.down3 = low_down(64, 128)
        self.down4 = low_down(128, 128)
        self.bottleneck_1 = low_double_conv(128, 128)
        self.bottleneck_2 = low_double_conv(128, 128)
        self.up1 = low_up(256, 128)
        self.up2 = low_up(192, 64)
        self.up3 = low_up(96, 32)
        self.up4 = low_up(48, 16)
        self.outc = nn.Sequential(nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0), nn.CELU())

        self.global_conv_1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.CELU()) #nn.InstanceNorm2d(128),
        self.global_conv_2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.CELU())
        self.global_conv_3 = nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.CELU()) #TODO upsampling trial
        self.global_conv_4 = nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.CELU())
        self.global_linear = nn.Linear(128*8**2, 128)


    def forward(self, x):
        #TODO serperatable normalize
        #input = torch.zeros(x.size(0), x.size(1)*2, x.size(2), x.size(3))
        normalized_input = torch.zeros_like(x)
        for i in range(x.size(0)):#batch size
            for j in range(x.size(1)):#channel depth: 3
                channel_mean = x[i][j].mean()
                cahnnel_mean_std = x[i][j].std()
                normalized_input[i][j] = (x[i][j] - channel_mean) / cahnnel_mean_std
        input = torch.cat([x, normalized_input], dim=1)
        x1 = self.inc(input)
        x2 = self.down1(x1)#1/2
        x3 = self.down2(x2)#1/4
        x4 = self.down3(x3)#1/8
        x5 = self.down4(x4)#1/16

        #global_feature = self.global_conv_1(x5)#1/32
        #global_feature = self.global_conv_2(global_feature) #1/64 128x8x8
        #global_feature = self.global_conv_3(global_feature)
        #global_feature = self.global_conv_4(global_feature) #1/16
        #TODO dynamic linear F.linear(global_feature.shape....) 이러면 학습이 되나 ? 
        #global_feature = global_feature.view(x.size(0), -1)
        #global_feature = self.global_linear(global_feature).view(x.size(0), -1, 1, 1)
        #global_feature = global_feature.expand_as(x5)

        x6 = self.bottleneck_1(x5) #1/16
        #x7 = self.bottleneck_2(torch.cat((x6, global_feature), dim=1))
        x7 = self.bottleneck_2(x6)
        out = self.up1(x7, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        out = self.outc(out)
        return out+x #res


class ResBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, padding=1, stride=1):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
                            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1),
                            nn.BatchNorm2d(dim),
                            nn.CELU(),
                            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1),
                            nn.BatchNorm2d(dim),
                            nn.CELU()
                            )

    def forward(self, x):
        return self.conv(x) + x

class BasicBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, padding=1, stride=1):
        super(BasicBlock, self).__init__()
        self.conv = nn.Sequential(
                            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1),
                            nn.BatchNorm2d(dim),
                            nn.CELU(),
                            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1),
                            nn.BatchNorm2d(dim),
                            nn.CELU()
                            )

    def forward(self, x):
        return self.conv(x)

class Low_ResBlock(nn.Module):
    def __init__(self, dim, low_dim=16, kernel_size=3, padding=1, stride=1):
        super(Low_ResBlock, self).__init__()
        self.conv = nn.Sequential(
                            nn.Conv2d(dim, low_dim, kernel_size=1, stride=1, padding=0),
                            nn.BatchNorm2d(low_dim),
                            nn.CELU(),
                            nn.Conv2d(low_dim, low_dim, kernel_size=3, stride=1, padding=1, groups=low_dim),
                            nn.BatchNorm2d(low_dim),
                            nn.CELU(),
                            nn.Conv2d(low_dim, dim, kernel_size=1, stride=1, padding=0),
                            nn.BatchNorm2d(dim),
                            nn.CELU(),
                            )

    def forward(self, x):
        return self.conv(x) + x

class Resnet_Generator(nn.Module):
    def __init__(self, ):
        super(Resnet_Generator, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.celu = nn.CELU()

        self.block1 = ResBlock(64)
        self.block2 = ResBlock(64)
        self.block3 = ResBlock(64)
        #self.block4 = ResBlock(64)
        #self.block5 = ResBlock(64)

        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        #TODO serperatable normalize
        #input = torch.zeros(x.size(0), x.size(1)*2, x.size(2), x.size(3))
        normalized_input = torch.zeros_like(x)
        for i in range(x.size(0)):#batch size
            for j in range(x.size(1)):#channel depth: 3
                channel_mean = x[i][j].mean()
                cahnnel_mean_std = x[i][j].std()
                normalized_input[i][j] = (x[i][j] - channel_mean) / cahnnel_mean_std
        input = torch.cat([x, normalized_input], dim=1)
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.celu(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        #out = self.block4(out)
        #out = self.block5(out)

        out = self.conv2(out)
        out = self.celu(out)

        return out+x

class Low_Resnet_Generator(nn.Module):
    def __init__(self, ):
        super(Low_Resnet_Generator, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.celu = nn.CELU()

        self.block1 = Low_ResBlock(64)
        self.block2 = Low_ResBlock(64)
        #self.block3 = Low_ResBlock(64)
        #self.block4 = ResBlock(64)
        #self.block5 = ResBlock(64)

        self.conv2 = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        #TODO serperatable normalize
        #input = torch.zeros(x.size(0), x.size(1)*2, x.size(2), x.size(3))
        normalized_input = torch.zeros_like(x)
        for i in range(x.size(0)):#batch size
            for j in range(x.size(1)):#channel depth: 3
                channel_mean = x[i][j].mean()
                cahnnel_mean_std = x[i][j].std()
                normalized_input[i][j] = (x[i][j] - channel_mean) / cahnnel_mean_std
        input = torch.cat([x, normalized_input], dim=1)
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.celu(out)
        out = self.block1(out)
        out = self.block2(out)
        #out = self.block3(out)
        #out = self.block4(out)
        #out = self.block5(out)

        out = self.conv2(out)
        out = self.celu(out)

        return out+x

class Simple_Generator(nn.Module):
    def __init__(self, ):
        super(Simple_Generator, self).__init__()
        self.layers = nn.Sequential(
                            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(64),
                            nn.SELU())
        self.block1 = BasicBlock(64)
        self.block2 = BasicBlock(64)
        self.block3 = BasicBlock(64)
        self.block4 = BasicBlock(64)
        self.block5 = BasicBlock(64)

        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.celu = nn.SELU()

    def forward(self, x):
        #TODO serperatable normalize
        #input = torch.zeros(x.size(0), x.size(1)*2, x.size(2), x.size(3))
        normalized_input = torch.zeros_like(x)
        for i in range(x.size(0)):#batch size
            for j in range(x.size(1)):#channel depth: 3
                channel_mean = x[i][j].mean()
                cahnnel_mean_std = x[i][j].std()
                normalized_input[i][j] = (x[i][j] - channel_mean) / cahnnel_mean_std
        input = torch.cat([x, normalized_input], dim=1)
        out = self.layers(input)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)

        out = self.conv2(out)
        out = self.celu(out)

        return out+x

class Hourglass_Generator(nn.Module):
    def __init__(self, ):
        super(Generator, self).__init__()
        
    def forward(self, x):
        #TODO serperatable normalize
        #input = torch.zeros(x.size(0), x.size(1)*2, x.size(2), x.size(3))
        normalized_input = torch.zeros_like(x)
        for i in range(x.size(0)):#batch size
            for j in range(x.size(1)):#channel depth: 3
                channel_mean = x[i][j].mean()
                cahnnel_mean_std = x[i][j].std()
                normalized_input[i][j] = (x[i][j] - channel_mean) / cahnnel_mean_std
        input = torch.cat([x, normalized_input], dim=1)
        

        return out+x

class Discriminator(nn.Module):
    def __init__(self, ):
        super(Discriminator, self).__init__()
        
        layers = [nn.Conv2d(3, 16, 3, 1, 1), nn.LeakyReLU(0.2), nn.Conv2d(16, 16, 3, 1, 1), nn.LeakyReLU(0.2), nn.AvgPool2d(2,2)]  #512->256 1/2
        layers += [nn.Conv2d(16, 32, 3, 1, 1), nn.LeakyReLU(0.2), nn.Conv2d(32, 32, 3, 1, 1), nn.LeakyReLU(0.2), nn.AvgPool2d(2,2)] #256->128 1/4
        layers += [nn.Conv2d(32, 64, 3, 1, 1), nn.LeakyReLU(0.2), nn.Conv2d(64, 64, 3, 1, 1), nn.LeakyReLU(0.2), nn.AvgPool2d(2,2)] #128->64 1/8
        layers += [nn.Conv2d(64, 128, 3, 1, 1), nn.LeakyReLU(0.2), nn.Conv2d(128, 128, 3, 1, 1), nn.LeakyReLU(0.2), nn.AvgPool2d(2,2)] # 1/16
        layers += [nn.Conv2d(128, 128, 3, 1, 1), nn.LeakyReLU(0.2), nn.Conv2d(128, 128, 3, 1, 1), nn.LeakyReLU(0.2), nn.AvgPool2d(2,2)] #1/32
        self.linear_1 = nn.Linear(128, 512)
        self.linear_2 = nn.Linear(512, 256)
        self.linear_3 = nn.Linear(256, 1)

        self.discriminator = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.discriminator(x)
        out = F.avg_pool2d(out, 16, stride=1)
        out = out.view(x.size(0), -1)
        out = self.linear_1(out)
        out = self.linear_2(out)
        out = self.linear_3(out)
        out = out.squeeze_(dim=1)

        return out

class Discriminator_for_triple(nn.Module):
    def __init__(self, ):
        super(Discriminator_for_triple, self).__init__()
        
        layers = [nn.Conv2d(3, 16, 3, 1, 1), nn.LeakyReLU(0.2), nn.AvgPool2d(2,2)]  #512->256 1/2
        layers += [nn.Conv2d(16, 32, 3, 1, 1), nn.LeakyReLU(0.2), nn.AvgPool2d(2,2)] #256->128 1/4
        layers += [nn.Conv2d(32, 64, 3, 1, 1), nn.LeakyReLU(0.2), nn.AvgPool2d(2,2)] #128->64 1/8
        layers += [nn.Conv2d(64, 128, 3, 1, 1), nn.LeakyReLU(0.2), nn.AvgPool2d(2,2)] # 1/16
        layers += [nn.Conv2d(128, 128, 3, 1, 1), nn.LeakyReLU(0.2), nn.AvgPool2d(2,2)] #1/32
        self.linear_1 = nn.Linear(128, 512)
        self.linear_2 = nn.Linear(512, 256)
        self.linear_3 = nn.Linear(256, 1)

        self.discriminator = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.discriminator(x)
        out = F.avg_pool2d(out, 16, stride=1)
        out = out.view(x.size(0), -1)
        out = self.linear_1(out)
        out = self.linear_2(out)
        out = self.linear_3(out)
        out = out.squeeze_(dim=1)

        return out


class test_Generator(nn.Module):
    def __init__(self, ):
        super(test_Generator, self).__init__()
        self.inc = double_conv(6, 16)
        self.down1 = test_down(16, 32)
        self.down2 = test_down(32, 64)
        self.down3 = test_down(64, 128)
        self.down4 = test_down(128, 128)
        self.bottleneck_1 = test_double_conv(128, 128)
        self.bottleneck_2 = test_double_conv(128, 128)
        self.up1 = test_up(256, 128)
        self.up2 = test_up(192, 64)
        self.up3 = test_up(96, 32)
        self.up4 = test_up(48, 16)
        self.outc = nn.Sequential(nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0), nn.CELU())


    def forward(self, x):
        #TODO serperatable normalize
        #input = torch.zeros(x.size(0), x.size(1)*2, x.size(2), x.size(3))
        normalized_input = torch.zeros_like(x)
        for i in range(x.size(0)):#batch size
            for j in range(x.size(1)):#channel depth: 3
                channel_mean = x[i][j].mean()
                cahnnel_mean_std = x[i][j].std()
                normalized_input[i][j] = (x[i][j] - channel_mean) / cahnnel_mean_std
        input = torch.cat([x, normalized_input], dim=1)
        x1 = self.inc(input)
        x2 = self.down1(x1)#1/2
        x3 = self.down2(x2)#1/4
        x4 = self.down3(x3)#1/8
        x5 = self.down4(x4)#1/16

        x6 = self.bottleneck_1(x5) #1/16
        x7 = self.bottleneck_2(x6)
        out = self.up1(x7, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        out = self.outc(out)
        return out+x #res


class HourGlass_Block(nn.Module):
    def __init__(self, in_plane, out_plane):
        super(HourGlass_Block, self).__init__()
        self.conv = nn.Sequential(
                            nn.MaxPool2d(2, 2),
                            nn.Conv2d(in_plane, out_plane, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(out_plane),
                            nn.CELU(inplace=True),
                            )
    def forward(self, x):
        out = self.conv(x)
        return out

class HourGlass_upBlock(nn.Module):
    def __init__(self, in_plane, out_plane):
        super(HourGlass_upBlock, self).__init__()
        self.conv = nn.Sequential(
                            nn.ConvTranspose2d(in_plane, out_plane, kernel_size=4, stride=2, padding=1),
                            nn.BatchNorm2d(out_plane),
                            nn.CELU(inplace=True),
                            )
    def forward(self, x):
        out = self.conv(x)
        return out

class HourGlass_skipBlock(nn.Module):
    def __init__(self, in_plane, out_plane, seperable):
        super(HourGlass_skipBlock, self).__init__()
        if seperable:
            self.conv = nn.Sequential(
                            nn.Conv2d(in_plane, out_plane, kernel_size=3, stride=1, padding=1, groups=in_plane),
                            nn.BatchNorm2d(out_plane),
                            nn.CELU(inplace=True),
                            )
        else:
            self.conv = nn.Sequential(
                            nn.Conv2d(in_plane, out_plane, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(out_plane),
                            nn.CELU(inplace=True),
                            )

    def forward(self, x):
        out = self.conv(x)
        return x

class StackedHourGlass(nn.Module):
    def __init__(self, seperable=False):
        super(StackedHourGlass, self).__init__()
        self.inconv = nn.Sequential(
                            nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(16),
                            nn.CELU(inplace=True)
                            )
        self.block_1 = HourGlass_Block(16, 32) #1/2
        self.block_2 = HourGlass_Block(32, 64) #1/4
        self.block_3 = HourGlass_Block(64, 128) #1/8
        self.block_4 = HourGlass_Block(128, 256) #1/16 =>(32x32)
        self.block_5 = HourGlass_Block(256, 512) #1/32

        self.bottleneck = nn.Sequential(
                                        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(512),
                                        nn.CELU(inplace=True)
                                        )

        self.up_block_1 = HourGlass_upBlock(512, 256)
        self.up_block_2 = HourGlass_upBlock(256, 128)
        self.up_block_3 = HourGlass_upBlock(128, 64)
        self.up_block_4 = HourGlass_upBlock(64, 32)
        self.up_block_5 = HourGlass_upBlock(32, 16)

        self.skip_block_1 = HourGlass_skipBlock(32, 32, seperable)
        self.skip_block_2 = HourGlass_skipBlock(64, 64, seperable)
        self.skip_block_3 = HourGlass_skipBlock(128, 128, seperable)
        self.skip_block_4 = HourGlass_skipBlock(256, 256, seperable)
        self.skip_block_5 = HourGlass_skipBlock(512, 512, seperable)

        self.outconv = nn.Sequential(
                                nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
                                nn.CELU(inplace=True),
                                )

    def forward(self, x):
        normalized_input = torch.zeros_like(x)
        for i in range(x.size(0)):#batch size
            for j in range(x.size(1)):#channel depth: 3
                channel_mean = x[i][j].mean()
                cahnnel_mean_std = x[i][j].std()
                normalized_input[i][j] = (x[i][j] - channel_mean) / cahnnel_mean_std

        input = torch.cat([x, normalized_input], dim=1)

        out_0 = self.inconv(input)
        out_1 = self.block_1(out_0) #32x256x256
        out_2 = self.block_2(out_1) #64x128x128
        out_3 = self.block_3(out_2) #128x64x64
        out_4 = self.block_4(out_3) #256x32x32
        out_5 = self.block_5(out_4) #512x16x16
        
        out_6 = self.bottleneck(out_5) #512x16x16
        
        skip_out1 = self.skip_block_1(out_1) #32x256x256
        skip_out2 = self.skip_block_2(out_2) #64x128x128
        skip_out3 = self.skip_block_3(out_3) #128x64x64
        skip_out4 = self.skip_block_4(out_4) #256x32x32
        skip_out5 = self.skip_block_5(out_5) #512x16x16

        out = self.up_block_1(out_6+skip_out5) #256x32x32
        out = self.up_block_2(out+skip_out4) #128x64x64
        out = self.up_block_3(out+skip_out3) #64x128x128
        out = self.up_block_4(out+skip_out2) #32x256x256
        out = self.up_block_5(out+skip_out1) #16x512x512
        out = self.outconv(out)
        return out + x

class Global_Unet(nn.Module):
    def __init__(self, normalized_concat=False):
        super(Global_Unet, self).__init__()

        self.normalized_concat = normalized_concat

        if normalized_concat:
            self.inc = inconv(6, 16)
        else:
            self.inc = inconv(3, 16)

        self.down1 = down(16, 32)
        self.down2 = down(32, 64)
        self.down3 = down(64, 128)
        self.down4 = down(128, 128)
        
        self.bottleneck_1 = double_conv(128, 128)
        self.bottleneck_2 = double_conv(256, 128)

        self.up1 = up(256, 128)
        self.up2 = up(192, 64)
        self.up3 = up(96, 32)
        self.up4 = up(48, 16)
        self.outc = nn.Sequential(nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0), nn.SELU())

        self.global_conv_1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.SELU()) #nn.InstanceNorm2d(128),
        self.global_conv_2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.SELU())
        self.global_linear = nn.Linear(128*8**2, 128)

    def forward(self, x):
        
        if self.normalized_concat:
            normalized_input = torch.zeros_like(x)
            for i in range(x.size(0)):#batch size
                for j in range(x.size(1)):#channel depth: 3
                    channel_mean = x[i][j].mean()
                    cahnnel_mean_std = x[i][j].std()
                    normalized_input[i][j] = (x[i][j] - channel_mean) / cahnnel_mean_std
            input = torch.cat([x, normalized_input], dim=1)
        else:
            input = x

        x1 = self.inc(input)
        x2 = self.down1(x1)#1/2
        x3 = self.down2(x2)#1/4
        x4 = self.down3(x3)#1/8
        x5 = self.down4(x4)#1/16

        global_feature = self.global_conv_1(x5)#1/32
        global_feature = self.global_conv_2(global_feature) #1/64 128x8x8
        global_feature = global_feature.view(x.size(0), -1)
        global_feature = self.global_linear(global_feature).view(x.size(0), -1, 1, 1)
        global_feature = global_feature.expand_as(x5)

        x6 = self.bottleneck_1(x5) #1/16
        x7 = self.bottleneck_2(torch.cat((x6, global_feature), dim=1))

        out = self.up1(x7, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        out = self.outc(out)
        return out+x #res

class Global_Unet_fully_convolution(nn.Module):
    def __init__(self, normalized_concat=False):
        super(Global_Unet_fully_convolution, self).__init__()

        self.normalized_concat = normalized_concat

        if normalized_concat:
            self.inc = inconv(6, 16)
        else:
            self.inc = inconv(3, 16)

        self.down1 = down(16, 32)
        self.down2 = down(32, 64)
        self.down3 = down(64, 128)
        self.down4 = down(128, 128)
        
        self.bottleneck_1 = double_conv(128, 128)
        self.bottleneck_2 = double_conv(256, 128)

        self.up1 = up(256, 128)
        self.up2 = up(192, 64)
        self.up3 = up(96, 32)
        self.up4 = up(48, 16)
        self.outc = nn.Sequential(nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0), nn.SELU())

        self.global_conv_1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128), nn.BatchNorm2d(128), nn.SELU()) #nn.InstanceNorm2d(128),
        self.global_conv_2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128), nn.BatchNorm2d(128), nn.SELU())
        #self.global_linear = nn.Linear(128*8**2, 128)

    def forward(self, x):
        
        if self.normalized_concat:
            normalized_input = torch.zeros_like(x)
            for i in range(x.size(0)):#batch size
                for j in range(x.size(1)):#channel depth: 3
                    channel_mean = x[i][j].mean()
                    cahnnel_mean_std = x[i][j].std()
                    normalized_input[i][j] = (x[i][j] - channel_mean) / cahnnel_mean_std
            input = torch.cat([x, normalized_input], dim=1)
        else:
            input = x

        x1 = self.inc(input)
        x2 = self.down1(x1)#1/2
        x3 = self.down2(x2)#1/4
        x4 = self.down3(x3)#1/8
        x5 = self.down4(x4)#1/16

        global_feature = self.global_conv_1(x5)#1/32
        global_feature = self.global_conv_2(global_feature) #1/64 128x8x8
        #global_feature = global_feature.view(x.size(0), -1)
        #global_feature = self.global_linear(global_feature).view(x.size(0), -1, 1, 1)
        #global_feature = global_feature.expand_as(x5)

        x6 = self.bottleneck_1(x5) #1/16
        x7 = self.bottleneck_2(torch.cat((x6, global_feature), dim=1))

        out = self.up1(x7, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        out = self.outc(out)
        return out+x #res



class Image_Quality_Enhance_model(nn.Module):
    def __init__(self, model='None'):
        super(Image_Quality_Enhance_model, self).__init__()
        if model=='unet':
            self.layers = Unet_Generator()
        elif model=='resnet':
            self.layers = Resnet_Generator()
        elif model=='simple':
            self.layers = Simple_Generator()
        elif model=='low-resnet':
            self.layers = Low_Resnet_Generator()
        elif model=='low-unet':
            self.layers = Low_Unet_Generator()
        elif model=='triple-unet':
            self.layers = Triple_Unet_Generator()
        elif model=='hourglass':
            self.layers = StackedHourGlass(seperable=True)
        elif model=='hourglass-seperable-false':
            self.layers = StackedHourGlass(seperable=False)
        elif model=='global-unet':
            self.layers = Global_Unet(normalized_concat=False)
        elif model=='global-unet-normalized_concat':
            self.layers = Global_Unet(normalized_concat=True)
        elif model=='global_unet_fully_convolution':
            self.layers = Global_Unet_fully_convolution(normalized_concat=False)
        elif model=='global_unet_fully_convolution-normalized-concat':
            self.layers = Global_Unet_fully_convolution(normalized_concat=True)
        elif model=='test':
            self.layers = test_Generator()
        else:
            pass
            
    def forward(self, x):
        out = self.layers(x)
        return out