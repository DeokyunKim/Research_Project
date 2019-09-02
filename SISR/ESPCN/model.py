import torch.nn as nn
import math

class ESPCN(nn.Module):
    """Single Image Super-resolutuion 

    Parameters:
        upscale_factor (int)    -- upscale_factor 2 | 4 | 8
    """
    def __init__(self, upscale_factor=4):
        super(ESPCN, self).__init__()
        
        sequence = [nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU()]
        nbr_of_stacked_layer = int(math.log(upscale_factor,2))

        ngf = 64 
        for i in range(nbr_of_stacked_layer): # if upscale factor is 8, will stack 3 layers. last layers' nbr of channel is 64*8 = 512 
            mult = 2 ** i
            sequence += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size = 3, stride=1, padding=1),
                         nn.ReLU()]
        #stacked pixel shuffle
        nbr_of_last_channel = ngf * mult * 2
        for i in range(nbr_of_stacked_layer): # 512, 128, 32, 8
            mult = 4 ** (i+1)
            channels = int(nbr_of_last_channel/mult) #feature map to feature map eqaully mapping. (nbr_channel -> nbr_channel)
            sequence += [nn.PixelShuffle(2),
                         nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                         nn.ReLU()]
        #to rgb
        sequence += [nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=1),
                     nn.Tanh()]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)

if __name__ == '__main__':
    print(ESPCN(8))