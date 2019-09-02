from dataloader import RGB_NIR_Dataset
import argparse
from torch.utils.data import DataLoader
from model import *
from torchvision import utils
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import math
import numpy as np
from torch.autograd import Variable, grad
import os
from ssim import SSIM
from utils import requires_grad

def train(epoch, dataloader, generator, discriminator, g_optim, d_optim, MSE_Loss, SSIM_Loss):
    avg_generator_loss = 0
    avg_discriminator_loss = 0
    for i, (rgb, nir) in enumerate(dataloader):
        target_image = rgb.to(device)
        real_nir = nir.to(device)
        fake_rgb = generator(real_nir)
        predict_fake = discriminator(fake_rgb)

        gan_loss = -predict_fake.mean()
        pixel_loss = MSE_Loss(fake_rgb, target_image)
        dssim_loss = (1-SSIM_Loss(fake_rgb, target_image))/2.

        generator_loss = 0.15*gan_loss + 0.65*pixel_loss + 0.2*dssim_loss

        g_optim.zero_grad()
        generator_loss.backward()
        g_optim.step()

        predict_real = discriminator(target_image)
        predict_real = predict_real.mean() - 0.001 * (predict_real ** 2).mean()
        fake_rgb = generator(real_nir)
        predict_fake = discriminator(fake_rgb)
        predict_fake = predict_fake.mean()

        #Gradient Penalty (GP)
        eps = torch.rand(real_nir.size(0), 1, 1, 1).to(device)
        x_hat = eps * target_image.data + (1-eps) * fake_rgb.data
        x_hat = Variable(x_hat, requires_grad=True).to(device)
        hat_predict = discriminator(x_hat)
        grad_x_hat = grad(
            outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
        grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) -1)**2).mean()
        #grad_penalty = torch.max(torch.zeros(1).to(device)
        #    , ((grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) -1)).mean())
        grad_penalty = 10 * grad_penalty
        
        d_loss = predict_fake - predict_real + grad_penalty

        d_optim.zero_grad()
        d_loss.backward()
        d_optim.step()        

        sys.stdout.write('\r Epoch: %d [%d/%d] g_loss %.4f d_loss %.4f'%(epoch, i, len(dataloader), generator_loss.item(), d_loss.item()))
        avg_generator_loss += generator_loss.item()
        avg_discriminator_loss += d_loss.item()

    print('\n Average generator loss: %.4f, Average discriminator loss: %.4f'%(avg_generator_loss/len(dataloader), avg_discriminator_loss/len(dataloader)))
    nir_image = torch.cat([real_nir, real_nir, real_nir], dim=1)
    imgs = torch.cat((nir_image, fake_rgb, target_image), 0)
    utils.save_image(imgs, os.path.join(args.result_path,'results.jpeg'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='from NIR to RGB conversion')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--gpu-ids', default=[0,1,2,3], type=int)
    parser.add_argument('--result-path', default='./results/', type=str)
    parser.add_argument('--checkpoint-path', default='./checkpoints/', type=str)
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.checkpoint_path):
        os.mkdir(args.checkpoint_path)
        print('===>make directory', args.checkpoint_path)
    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)
        print('===>make directory', args.result_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = RGB_NIR_Dataset(mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    generator = RGB_Generator().to(device)
    discriminator = Discriminator().to(device)

    g_optim = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.,0.99))
    d_optim = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0., 0.99))

    MSE_Loss = nn.MSELoss().to(device)
    ssim_loss = SSIM()
    for i in range(args.epochs):
        train(i, dataloader, generator, discriminator, g_optim, d_optim, MSE_Loss, ssim_loss)
        print('epoch %3d is done'%i)
