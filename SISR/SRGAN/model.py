import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super().__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)


class BasicBlock(nn.Module):
    """Generate Dense Block
    Parameters:
        in_channels (int)   -- number of input channels
        out_channels (int)  -- number of output channels
        kernel (int)        -- kernel
        stride (int)        -- stride
        padding (int)       -- padding

    """
    def __init__(self, in_channels, kernel_size, stride, padding):
        super(BasicBlock, self).__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                                    nn.BatchNorm2d(in_channels),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                                    nn.BatchNorm2d(in_channels),
                                    nn.ReLU())
    def forward(self, x):
        out = self.layers(x)
        return torch.cat([x, out], dim=1)

class DenseBlock(nn.Module):
    """Generate Dense Block
    Parameters:
        nbr_of_layers (int) -- number of layers
        in_channels (int)   -- number of input channels
        block (nn.Module)   -- stacked block (BasicBlock)
    """
    def __init__(self, nbr_of_layers, ngf=64, in_channels=3):
        super(DenseBlock, self).__init__()
        sequence = [nn.Conv2d(in_channels, ngf, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(ngf),
                    nn.ReLU()]

        for i in range(nbr_of_layers):
            sequence += [BasicBlock(ngf*(2**i), kernel_size=3, stride=1, padding=1)]
        self.layers = nn.Sequential(*sequence)

    def forward(self, x):
        return self.layers(x)


class Generator(nn.Module):
    """Single Image Super-resolutuion Generator based on DenseBlock

    Parameters:
        upscale_factor (int)    -- upscale_factor 2 | 4 | 8
    """
    def __init__(self, upscale_factor=4, ngf=128):
        super(Generator, self).__init__()
        self.upscale_factor = upscale_factor
        nbr_of_layers = int(math.log(upscale_factor, 2))
        model = DenseBlock(nbr_of_layers, ngf, in_channels=3)
        self.model = nn.Sequential(model, 
                              nn.PixelShuffle(upscale_factor),
                              nn.Conv2d(int(ngf*(2**(nbr_of_layers))/(upscale_factor**2)), 3, kernel_size=3, stride=1, padding=1),
                              nn.Tanh())
    def forward(self, x):
        out = self.model(x)
        return out #+ F.interpolate(x, scale_factor=self.upscale_factor, mode='bilinear')

class Discriminator(nn.Module):
    def __init__(self,):
        super().__init__()
        self.layers = nn.Sequential(
                            nn.Conv2d(3, 64, 3, 1, 1),
                            nn.BatchNorm2d(64),
                            nn.LeakyReLU(0.2),
                            nn.MaxPool2d(kernel_size=2, stride=2),
                            nn.Conv2d(64, 128, 3, 1, 1),
                            nn.BatchNorm2d(128),
                            nn.LeakyReLU(0.2),
                            nn.MaxPool2d(kernel_size=2, stride=2),
                            nn.Conv2d(128, 256, 3, 1, 1),
                            nn.BatchNorm2d(256),
                            nn.LeakyReLU(0.2),
                            nn.MaxPool2d(kernel_size=2, stride=2),
                            nn.Conv2d(256, 1, 1, 1, 1)
        )
    def forward(self, x):
        out = self.layers(x)
        out = F.sigmoid(F.avg_pool2d(out, out.size()[2:])).view(out.size(0), -1)
        return out

if __name__ == '__main__':
    print(Generator(upscale_factor=8))