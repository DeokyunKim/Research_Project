import torch.nn as nn
import torch
from torch.nn import functional as F

class channel_layers(nn.Module):
    def __init__(self, in_plane=1, nfg= 64, kernel_size=3, stride=1, padding=1):
        super(channel_layers, self).__init__()
        layers = [nn.Conv2d(in_plane, nfg, kernel_size=kernel_size, stride=stride, padding=padding), nn.InstanceNorm2d(64), nn.SELU()]
        for i in range(3):
            layers += [nn.MaxPool2d(2, 2),
                        nn.Conv2d(nfg*(2**i), nfg*(2**(i+1)), kernel_size=kernel_size, stride=stride, padding=padding), 
                        nn.InstanceNorm2d(nfg*(2**(i+1))),
                        nn.SELU(),
                        nn.Conv2d(nfg*(2**(i+1)), nfg*(2**(i+1)), kernel_size=kernel_size, stride=stride, padding=padding),
                        nn.InstanceNorm2d(nfg*(2**(i+1))),
                        nn.SELU()]

        for i in range(3):
            in_channel = nfg*(2**(3-i))
            layers += [nn.Upsample(scale_factor=2, mode='bilinear'),
                         nn.Conv2d(in_channel, int(in_channel//2), kernel_size=kernel_size, stride=stride, padding=padding),
                         nn.InstanceNorm2d(int(in_channel//2)), 
                         nn.SELU(),
                         nn.Conv2d(in_channel//2, int(in_channel//2), kernel_size=kernel_size, stride=stride, padding=padding),
                         nn.InstanceNorm2d(int(in_channel//2)), 
                         nn.SELU()]

        layers += [nn.Conv2d(nfg, 1, kernel_size=kernel_size, stride=stride, padding=padding), nn.Tanh()]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x) + x 

class BasicBlock(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size=3, stride=1, padding=1):
        super(BasicBlock, self).__init__()

        layers = [nn.Conv2d(in_plane, out_plane, kernel_size=kernel_size, stride=stride, padding=padding), nn.InstanceNorm2d(out_plane), nn.ReLU()]
        layers += [nn.Conv2d(out_plane, out_plane, kernel_size=kernel_size, stride=stride, padding=padding), nn.InstanceNorm2d(out_plane), nn.ReLU()]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class NIR_to_RGB(nn.Module):
    def __init__(self,):
        super(NIR_to_RGB, self).__init__()
        self.nir_to_R = channel_layers(in_plane=1, nfg=32, kernel_size=3, stride=1, padding=1)
        self.nir_to_G = channel_layers(in_plane=1, nfg=32, kernel_size=3, stride=1, padding=1)
        self.nir_to_B = channel_layers(in_plane=1, nfg=32, kernel_size=3, stride=1, padding=1)
        
        """
        model = [BasicBlock(in_plane=1, out_plane=32),
                BasicBlock(in_plane=32, out_plane=64),
                BasicBlock(in_plane=64, out_plane=128),
                BasicBlock(in_plane=128, out_plane=64),
                BasicBlock(in_plane=64, out_plane=32),
                nn.Conv2d(32, 3, 3, 1, 1),
                nn.ReLU()
                ]
        self.model = nn.Sequential(*model)
        """
    def forward(self, x):
        R = self.nir_to_R(x)
        G = self.nir_to_G(x)
        B = self.nir_to_B(x)
        colorized_image = torch.cat([R,G,B], dim=1)
        #nir_image = torch.cat([x,x,x], dim=1)
        #colorized_image = self.model(x)
        #print(colorized_image.shape)
        return colorized_image

class channelwise_generator(nn.Module):
    def __init__(self, residual=False):
        super(channelwise_generator, self).__init__()
        self.layers = nn.Sequential(
                            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
                            nn.BatchNorm2d(16),
                            nn.ReLU(),
                            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(),
                            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(),
                            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(16),
                            nn.ReLU(),
                            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
                            nn.ReLU())
        self.residual = residual

    def forward(self, x):
        if self.residual:
            return x+self.layers(x)
        else:
            return self.layers(x)

class RGB_Generator(nn.Module):
    def __init__(self, ):
        super(RGB_Generator, self).__init__()
        self.R_layer = channelwise_generator(residual=True)
        self.G_layer = channelwise_generator(residual=False)
        self.B_layer = channelwise_generator(residual=False)

    def forward(self, x):
        R = self.R_layer(x)
        G = self.G_layer(x)
        B = self.B_layer(x)
        out = torch.cat([R, G, B], dim=1)
        return out


class Discriminator(nn.Module):
    def __init__(self, in_plane=3):
        super(Discriminator, self).__init__()

        layers = [nn.Conv2d(in_plane, 32, 3, 1, 1), nn.LeakyReLU(negative_slope=0.05), nn.Conv2d(32, 32, 3, 1, 1), nn.LeakyReLU(negative_slope=0.05), nn.AvgPool2d(2,2)]  #128->64
        layers += [nn.Conv2d(32, 64, 3, 1, 1), nn.LeakyReLU(negative_slope=0.05), nn.Conv2d(64, 64, 3, 1, 1), nn.LeakyReLU(negative_slope=0.05), nn.AvgPool2d(2,2)] #64->32
        layers += [nn.Conv2d(64, 128, 3, 1, 1), nn.LeakyReLU(negative_slope=0.05), nn.Conv2d(128, 128, 3, 1, 1), nn.LeakyReLU(negative_slope=0.05), nn.AvgPool2d(2,2)] #32->16
        layers += [nn.Conv2d(128, 256, 3, 1, 1), nn.LeakyReLU(negative_slope=0.05), nn.Conv2d(256, 256, 3, 1, 1), nn.LeakyReLU(negative_slope=0.05), nn.AvgPool2d(2,2)] #16->8 
        
        self.linear = nn.Linear(256, 512)
        self.linear2 = nn.Linear(512, 1)

        self.discriminator = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.discriminator(x)
        out = F.avg_pool2d(out, 16, stride=1)
        out = out.view(x.size(0), -1)
        out = self.linear(out)
        out = self.linear2(out)
        out = out.squeeze_(dim=1)

        return out

class double_conv(nn.Module):
    def __init__(self, in_plane, out_plane):
        super(double_conv, self).__init__()
        self.layers = nn.Sequential(
                            nn.Conv2d(in_plane, out_plane, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(out_plane),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(out_plane, out_plane, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(out_plane),
                            nn.ReLU(inplace=True),
                            )
    def forward(self, x):
        return self.layers(x)

class down(nn.Module):
    def __init__(self, in_plane, out_plane):
        super(down, self).__init__()
        self.layers = nn.Sequential(
                            nn.MaxPool2d(2, 2),
                            double_conv(in_plane, out_plane)
                            )
    def forward(self, x):
        return self.layers(x)

class up(nn.Module):
    def __init__(self, in_plane, out_plane):
        super(up, self).__init__()
        self.layers = nn.Sequential(
                            #nn.Upsample(scale_factor=2, mode='bilinear'),
                            nn.ConvTranspose2d(in_plane, out_plane, kernel_size=4, stride=2, padding=1),
                            double_conv(out_plane, out_plane)
                            )
    def forward(self, x):
        return self.layers(x)

class Embeder(nn.Module):
    def __init__(self, in_plane):
        super(Embeder, self).__init__()

        self.inconv = double_conv(in_plane, 16)

        self.down_1 = down(16, 32)
        self.down_2 = down(32, 64)
        self.down_3 = down(64, 128)
        self.down_4 = down(128, 256)

    def forward(self, x):
        out = self.inconv(x)
        out = self.down_1(out)
        out = self.down_2(out)
        out = self.down_3(out)
        out = self.down_4(out)
        return out


class Translator(nn.Module): 
    def __init__(self, in_plane=256, out_plane=1):
        super(Translator, self).__init__()
        self.up_1 = up(in_plane, in_plane//2)
        self.up_2 = up(in_plane//2, in_plane//4)
        self.up_3 = up(in_plane//4, in_plane//8)
        self.up_4 = up(in_plane//8, in_plane//16)

        self.out_conv = double_conv(in_plane//16, out_plane)

    def forward(self, x):
        out = self.up_1(x)
        out = self.up_2(out)
        out = self.up_3(out)
        out = self.up_4(out)
        out = self.out_conv(out)
        return out


class Embed_Translator(nn.Module):
    def __init__(self, in_plane, middle_plane=256):
        super(Embed_Translator, self).__init__()
        self.embed = Embeder(in_plane)
        self.inverse_embed = Translator(middle_plane, in_plane)

    def forward(self, x):
        z = self.embed(x)
        x_prime = self.inverse_embed(z)
        return z, x_prime



class NIR_AutoEncoder(nn.Module):
    def __init__(self, ):
        super(NIR_AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
                            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
                            nn.BatchNorm2d(16),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2, stride=2),#128
                            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2, stride=2),#64
                            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                            )

        self.decoder = nn.Sequential(
                            nn.Upsample(scale_factor=2, mode='bilinear'),
                            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(),
                            nn.Upsample(scale_factor=2, mode='bilinear'),
                            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(16),
                            nn.ReLU(),
                            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            )
    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out


