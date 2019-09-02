import torch.nn as nn
from torchvision import models
from collections import namedtuple

class Vgg16(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.layer1 = nn.Sequential()
        self.layer2 = nn.Sequential()
        self.layer3 = nn.Sequential()
        self.layer4 = nn.Sequential()

        for x in range(4):
            self.layer1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4,9):
            self.layer2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.layer3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.layer4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.layer1(x)
        h_relu_1_2 = h
        h = self.layer2(h)
        h_relu_2_2 = h
        h = self.layer3(h)
        h_relu_3_3 = h
        h = self.layer4(h)
        h_relu_4_3 = h

        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3
