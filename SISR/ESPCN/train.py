import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as utils
from dataloader import Dataset
from torch.utils.data import DataLoader
from model import ESPCN, FeatureExtractor
from math import log10

#train
def train(dataloader, model, epoch, criterion, optimizer, args):
    avg_loss = 0
    for i, (input_image, target_image) in enumerate(dataloader):
        input_image = input_image.to(device)
        target_image = target_image.to(device)
        output = model(input_image)
        loss = criterion(output, target_image)
        avg_loss += loss.item()
        sys.stdout.write('\r [%d][%d] Loss %.5f'%(i,len(dataloader), loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = avg_loss/len(dataloader)
    print('\n epoch [%d][%d] %.5f'%(epoch, args.epochs, avg_loss))

#evaluate
def evaluate(dataloader, model, epoch, criterion, args):
    avg_psnr = 0
    for i, (input_image, target_image) in enumerate(dataloader):
        input_image = input_image.cuda()
        target_image = target_image.cuda()
        output = model(input_image)
        psnr = 10*log10(1/criterion(output,target_image).item())
        avg_psnr += psnr
    avg_psnr = avg_psnr/len(dataloader)
    sys.stdout.write('\r\nPSNR [%d][%d] %.5fdB\r\n'%(epoch, args.epochs, avg_psnr))
    return avg_psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch Implementation ESPCN')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--path', default='./dataset/', type=str)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--upscale_factor', default=4, type=int)
    parser.add_argument('--gpu-ids', default=[0,1,2,3], nargs='+')
    parser.add_argument('--save-path', default='./checkpoints/', type=str)
    args = parser.parse_args()
    print(args)

    #mkdir
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    print('==>Loading Training Dataset')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Dataset(path=args.path, mode = 'train', upscale_factor=args.upscale_factor)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_dataset = Dataset(path=args.path, mode='val', upscale_factor=args.upscale_factor)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("==>Loading model")
    model = ESPCN(upscale_factor=args.upscale_factor).to(device)
    #data parallel
    if len(args.gpu_ids) > 1:
        model = nn.DataParallel(model, args.gpu_ids)
    
    print(model)

    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    criterion = nn.MSELoss().to(device)
    psnr_best = 0
    for epoch in range(args.epochs):
        train(train_dataloader, model, epoch, criterion, optimizer, args)
        psnr = evaluate(val_dataloader, model, epoch, criterion, args)
        if psnr > psnr_best:
            psnr_best = psnr
            save_path_name = os.path.join(args.save_path,'x{}_best_model.pth'.format(args.upscale_factor))
            print('model is saved at ', save_path_name)
            torch.save(model, save_path_name)