import torch
import argparse
import os
from dataloader import Fivek_Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable, grad
import torch.optim as optim
from model import *
import sys
from torchvision import utils, transforms
from tensorboardX import SummaryWriter
import time
from PIL import Image
from div2k_dataloader import Dataset
from voc_dataloader import VOC_dataset
import math
import time
import torch.nn.functional as F

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch implementation of Image qaulity enhancement using fivek dataset')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--result-path', default='./samsung/', type=str)
    parser.add_argument('--checkpoint-path', default='./checkpoints/', type=str)
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)
        print('===>make directory', args.result_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Fivek_Dataset(mode='test')
    #dataset = Dataset(path='../SISR/ESPCN/dataset')
    #dataset = VOC_dataset()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    #x is input image, y is target image
    generator = torch.load(os.path.join(args.checkpoint_path, 'best_generator_global_unet_fully_convolution-normalized-concat_20.183563.ckpt'), map_location=torch.device('cuda'))
    generator = generator.to(device)
    generator.eval()
    for param in generator.parameters():
        param.requires_grad = False
    mse_Loss = nn.MSELoss()
    avg_psnr = 0 
    """
    for i, (img_name, input_image) in enumerate(dataloader):
        t0 = time.time()
        input_image = input_image.to(device)
        input_image = F.upsample(input_image, size=(input_image.shape[2]//8*8, input_image.shape[3]//8*8))
        fake_image = generator(input_image)
        t1 = time.time()
        utils.save_image(fake_image, os.path.join(args.result_path, img_name[0]))

        sys.stdout.write('\r [%d/%d] FPS:%.4f progress....'%(i, len(dataloader),1/(t1-t0)))


    """
    for i, (input_image, target_image) in enumerate(dataloader):
        
        input_image = input_image.to(device)
        target_image = target_image.to(device)

        fake_image = generator(input_image)
        psnr = 10*math.log10(1/mse_Loss(fake_image, target_image).item())
        avg_psnr += psnr
        utils.save_image(input_image, os.path.join(args.result_path, 'inputs_%d.jpeg'%i))
        utils.save_image(fake_image, os.path.join(args.result_path, 'results_%d.jpeg'%i))
        utils.save_image(target_image, os.path.join(args.result_path, 'targets_%d.jpeg'%i))

        sys.stdout.write('\r [%d/%d] PSNR: %.4f dB test progress....'%(i, len(dataloader), psnr))
    print('Average PSNR is %.4f'%(avg_psnr/len(dataloader)))
    
    """
    file_name = 'KakaoTalk_20190624_203537702.jpg'
    image = Image.open(file_name)
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)
    fake_image = generator(image)
    utils.save_image(fake_image, 'results.jpeg')
    """