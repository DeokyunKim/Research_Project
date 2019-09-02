import torch
import argparse
import os
#from dataloader import Fivek_Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable, grad
import torch.optim as optim
from model import *
import sys
from torchvision import utils
#from tensorboardX import SummaryWriter
import time
import cv2
from torchvision import transforms
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch implementation of Image qaulity enhancement using fivek dataset')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--result-path', default='./results/', type=str)
    parser.add_argument('--checkpoint-path', default='./checkpoints/', type=str)
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #dataset = Fivek_Dataset(mode='test')
    #dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    #x is input image, y is target image
    generator = torch.load(os.path.join(args.checkpoint_path, 'best_generator_unet_21.254858.ckpt')).to(device)
    generator.eval()

    cap = cv2.VideoCapture(0)
    transform = transforms.ToTensor()
    if (cap.isOpened() == False):
        print("Camera can not be used")

    while(cap.isOpened()):#(True):
        t0 = time.time()
        ret, frame = cap.read()
        #cv2.imshow('frame', frame)
        image = frame
        frame = frame.astype('float')/255.0
        scale = torch.Tensor([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
        x = torch.from_numpy(frame.transpose(2, 0, 1)).type('torch.cuda.FloatTensor')
        x = x.unsqueeze(0).to(device)
        y = generator(x)
        y = y.squeeze(0)
        y = torch.clamp(y, min=0., max=1)*255.0
        y = y.detach().cpu().data.numpy().transpose(1, 2, 0)
        cv2.imshow('frame', y.astype('uint8'))
        #cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
