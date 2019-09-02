import sys
import torch
import torch.nn as nn
import argparse
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as utils
from torch.autograd import Variable
from dataloader import Dataset
from torch.utils.data import DataLoader
from vgg import Vgg16
import os

from math import log10
from model import Generator, Discriminator, FeatureExtractor
import torch.optim as optim

def train(dataloader, epoch, args):
    avg_loss = 0
    for i, (input_image, target_image) in enumerate(dataloader):
        input_image = input_image.cuda()
        target_image = target_image.cuda()
        output = generator(input_image)
        feature_real = vgg(target_image)
        feature_fake = vgg(output)
        target_real = Variable(torch.rand(input_image.size(0), 1)*0.5 + 0.7).cuda()
        target_fake = Variable(torch.rand(input_image.size(0), 1)*0.3).cuda()
        adversarial_loss = adversarial_criterion(discriminator(output), target_real)

        perceptual_loss = criterion(feature_fake.relu2_2, feature_real.relu2_2)
        loss = criterion(output, target_image)
        total_loss = loss + 1e-1*perceptual_loss + 1e-3*adversarial_loss
        avg_loss += total_loss.item()

        g_optim.zero_grad()
        loss.backward()
        g_optim.step()

        discriminator_loss = adversarial_criterion(discriminator(target_image), target_real) + \
                                adversarial_criterion(discriminator(Variable(output.data)), target_fake)
        sys.stdout.write('\r [%d][%d] Generator Loss %.5f Discriminator loss %.5f'%(i, len(dataloader), total_loss.item(), discriminator_loss.item()))
        
        d_optim.zero_grad()
        discriminator_loss.backward()
        d_optim.step()


def evaluate(dataloader, epoch, args):
    avg_psnr = 0
    for i, (input_image, target_image) in enumerate(dataloader):
        input_image = input_image.cuda()
        target_image = target_image.cuda()
        output = generator(input_image)
        psnr = 10*log10(1/criterion(output, target_image).item())
        avg_psnr += psnr
    avg_psnr = avg_psnr/len(dataloader)
    sys.stdout.write('\r\n psnr [%d][%d] %.5f\r\n'%(epoch, args.epochs, avg_psnr))
    
    #img save 
    imgs = torch.cat([output, target_image], dim=0)
    img_path = os.path.join(args.sample_save_path, 'results.jpeg')
    utils.save_image(imgs, img_path)
    print('validation sample is saved at ',img_path)

    return avg_psnr

if __name__=='__main__':
    parser = argparse.ArgumentParser('Pytorch Implementation SRGAN')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--path', default='../ESPCN/dataset/', type=str)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--upscale-factor', default=4, type=int)
    parser.add_argument('--gpu-ids', default=[0, 1, 2, 3], nargs='+')
    parser.add_argument('--save-path', default='./checkpoints/', type=str)
    parser.add_argument('--sample-save-path', default='./imgs/', type=str)
    args = parser.parse_args()
    print(args)
	
	#mkdir
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(args.sample_save_path):
        os.mkdir(args.sample_save_path)

    print('==>Loading Training Dataset')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Dataset(path=args.path, mode = 'train', upscale_factor=args.upscale_factor)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_dataset = Dataset(path=args.path, mode='val', upscale_factor=args.upscale_factor)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("==>Loading model")
    generator = Generator(upscale_factor=args.upscale_factor).to(device)
    #print(generator)
    discriminator = Discriminator().to(device)
    vgg = Vgg16(requires_grad=False).to(device)

    if len(args.gpu_ids) > 1:
        generator = nn.DataParallel(generator, args.gpu_ids)
        discriminator = nn.DataParallel(discriminator, args.gpu_ids)
        #TODO vgg dataparallel
        #vgg = nn.DataParallel(vgg, args.gpu_ids)

    g_optim = optim.Adam(generator.parameters(), lr=args.lr)
    d_optim = optim.Adam(discriminator.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    adversarial_criterion = nn.BCELoss()
    psnr_best = 0

    for epoch in range(args.epochs):
        train(train_dataloader, epoch, args)
        psnr = evaluate(val_dataloader, epoch, args)
        if psnr > psnr_best:
            psnr_best = psnr
            save_path_name = os.path.join(args.save_path,'x{}_best_model.pth'.format(args.upscale_factor))
            print('model is saved at ', save_path_name)
            torch.save(generator, save_path_name)