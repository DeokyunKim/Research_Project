import torch
import argparse
import os
from dataloader import Fivek_Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable, grad
import torch.optim as optim
from model import *
import sys
from torchvision import utils
from tensorboardX import SummaryWriter
import time
from utils import GaussianSmoothing
from vggnet import Vgg16
import math

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0.01)

def train(generator, discriminator, dataloader):
    avg_pixel_loss = 0
    avg_gan_loss = 0
    best_psnr = 0
    generator.train()
    for i, (input_image, target_image) in enumerate(dataloader):
        input_image = input_image.to(device)
        target_image = target_image.to(device)

        fake_image = generator(input_image)
        pixel_loss = MSE_Loss(fake_image, target_image)
        #identity_loss = MSE_Loss(fake_image, input_image)
            
        tv_loss = torch.sum(torch.abs(fake_image[:, :, :, 1:] - fake_image[:, :, :, :-1])) + \
                        torch.sum(torch.abs((fake_image[:, :, :1 :] - fake_image[:, :, -1:, :])))
        tv_loss = tv_loss.mean()
        avg_pixel_loss += pixel_loss.item()
            
        predict_fake = discriminator(fake_image)
        smoothing_fake_image = F.pad(fake_image, (2, 2, 2, 2), mode='reflect')
        smoothing_fake_image = smoothing(smoothing_fake_image)
        smoothing_target_image = F.pad(target_image, (2, 2, 2, 2), mode='reflect')
        smoothing_target_image = smoothing(smoothing_target_image)
        color_loss = MSE_Loss(smoothing_fake_image, smoothing_target_image)

        feature_real = vgg(target_image)
        feature_fake = vgg(fake_image)
        perceptual_loss=0
        for fr, ff in zip(feature_real, feature_fake):
            perceptual_loss += MSE_Loss(ff, fr)
        g_loss = 10*pixel_loss - 1e-3*predict_fake.mean() + 1e-3*perceptual_loss + 1e-7*tv_loss + 1e-3*color_loss #+ 1e-3*perceptual_loss #+ 1e3*identity_loss
            
            
        g_optim.zero_grad()
        g_loss.backward()
        g_optim.step()
            
        predict_real = discriminator(target_image)
        predict_real = predict_real.mean() - 0.001 * (predict_real ** 2).mean()
        fake_image = generator(input_image)
        predict_fake = discriminator(fake_image)
        predict_fake = predict_fake.mean()

        #Gradient Penalty (GP)
        eps = torch.rand(input_image.size(0), 1, 1, 1).to(device)
        x_hat = eps * target_image.data + (1-eps) * fake_image.data
        x_hat = Variable(x_hat, requires_grad=True).to(device)
        hat_predict = discriminator(x_hat)
        grad_x_hat = grad(
                    outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
        #Original W-GAN
        grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) -1)**2).mean()
        #grad_penalty = torch.max(torch.zeros(1).to(device), ((grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) -1)).mean())
        grad_penalty = 10 * grad_penalty
        d_loss = predict_fake - predict_real + grad_penalty

        d_optim.zero_grad()
        d_loss.backward()
        d_optim.step()

        sys.stdout.write('\r epoch:%3d Iteration:[%5d/%5d] Generator loss:%6.4f Discriminator losses: %6.4f %6.4f %6.4f %6.4f'%(epoch, i, len(dataloader), g_loss.item(), predict_fake.item(), predict_real.item(), grad_penalty.item(), d_loss.item()))
        #sys.stdout.write('\r Iteration:%5d Generator loss %6.4f'%(i, g_loss.item()))    
        time_ = time.time()
        writer.add_scalar('Generator_loss_unet', g_loss.item(), epoch*len(dataloader)+i, time_)
        writer.add_scalar('Discriminator_loss_unet', d_loss.item(), epoch*len(dataloader)+i, time_)
            

        if i%10==0:
            imgs = torch.cat((target_image, fake_image, input_image), 0)
            imgs = utils.make_grid(imgs, nrow=4)
            utils.save_image(imgs, os.path.join(args.result_path,'results_%s.jpeg'%args.model))


def eval(generator, discriminator, dataloader, best_psnr):
    avg_psnr = 0
    generator.eval()
    for i, (input_image, target_image) in enumerate(val_dataloaer):
        input_image = input_image.to(device)
        target_image = target_image.to(device)

        fake_image = generator(input_image)
        pixel_loss = MSE_Loss(fake_image, target_image)
        avg_psnr += 10 * math.log10(1/pixel_loss.item())
        sys.stdout.write('\r epoch:%d [%d/%d] validation progress...'%(epoch, i, len(val_dataloaer)))

    avg_psnr = avg_psnr / len(val_dataloaer)

    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(generator, os.path.join(args.checkpoint_path, 'best_generator_%s_%4f.ckpt'%(args.model, best_psnr)))
        print('best model saved PSNR:%f dB'%best_psnr)
    else:
        print('validation PSNR:%f dB'%avg_psnr)

    return best_psnr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch implementation of Image qaulity enhancement using fivek dataset')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--gpu-ids', default=[0,1,2,3], type=int, nargs='+')
    parser.add_argument('--result-path', default='./results/', type=str)
    parser.add_argument('--checkpoint-path', default='./checkpoints/', type=str)
    parser.add_argument('--model', default='global_unet_fully_convolution-normalized-concat', type=str, help='unet, resnet, simple, etc...')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.checkpoint_path):
        os.mkdir(args.checkpoint_path)
        print('===>make directory', args.checkpoint_path)
    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)
        print('===>make directory', args.result_path)

    writer = SummaryWriter(comment='model %s, learning rate %.4f, batch_size %d'%(args.model, args.lr, args.batch_size))


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Fivek_Dataset(mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    val_dataset = Fivek_Dataset(mode='val')
    val_dataloaer = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    generator = Image_Quality_Enhance_model(model=args.model).to(device)
    discriminator = Discriminator().to(device)
    smoothing = GaussianSmoothing(channels=3, kernel_size=5, sigma=1).to(device)
    vgg = Vgg16().to(device)
    if len(args.gpu_ids) > 1:
        generator= nn.DataParallel(generator, device_ids=args.gpu_ids)
        discriminator= nn.DataParallel(discriminator, device_ids=args.gpu_ids)
        vgg = nn.DataParallel(vgg, device_ids=args.gpu_ids)
        smoothing = nn.DataParallel(smoothing, device_ids=args.gpu_ids)
    
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    

    g_optim = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.,0.99))
    d_optim = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0., 0.99))
    MSE_Loss = nn.MSELoss().to(device)

    best_psnr=0
    for epoch in range(args.epochs):
        train(generator, discriminator, dataloader)
        print()
        best_psnr = eval(generator, discriminator, val_dataloaer, best_psnr)

       
    writer.close()
