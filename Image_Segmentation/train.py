from voc_dataloader import VOC_Dataset
from model import Segmentation_Model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import sys
import numpy as np
from torchvision import utils

import pdb


def cross_entropy2d(input, target):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input_ = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(input_, target) #+ 1e-2*torch.abs(torch.argmax(input, dim=1).view(-1) - target)

    return loss

def train(epoch, dataloader, model):
    avg_loss = 0
    for i, (img, ann) in enumerate(dataloader):
        img = img.to(device)
        ann = ann.to(device)

        pred = model(img)
        loss = cross_entropy2d(pred, ann)
        avg_loss += loss.item()
        optim.zero_grad()
        loss.backward()
        optim.step()

        sys.stdout.write('\r Epoch: %3d [%d/%d] Loss: %8.4f'%(epoch, i, len(dataloader), loss.item()))
        pred_rgb, rgb_t = decode_pred(pred, ann)
        utils.save_image(torch.cat([img, pred_rgb/255., rgb_t/255.], dim=0), 'results/results_%d.jpeg'%i)
    print('\n Average Loss:%4.3f'%(avg_loss/len(dataloader)))
    torch.save(model, 'checkpoints/model_%d.ckpt'%epoch)

def decode_pred(pred, target):
    pred = torch.argmax(pred, dim=1)
    target_mask = target.detach().cpu().numpy()
    label_mask = pred.detach().cpu().numpy()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    rt = target_mask.copy()
    gt = target_mask.copy()
    bt = target_mask.copy()
    for ll in range(0, 21):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]

        rt[target_mask == ll] = label_colours[ll, 0]
        gt[target_mask == ll] = label_colours[ll, 1]
        bt[target_mask == ll] = label_colours[ll, 2]

    """
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    """
    r = np.expand_dims(r, axis=1)
    g = np.expand_dims(g, axis=1)
    b = np.expand_dims(b, axis=1)
    rgb = np.concatenate((r, g, b), axis=1)
    rgb = torch.from_numpy(rgb)
    #rgb = rgb/255.0

    rt = np.expand_dims(rt, axis=1)
    gt = np.expand_dims(gt, axis=1)
    bt = np.expand_dims(bt, axis=1)
    rgbt = np.concatenate((rt, gt, bt), axis=1)
    rgbt = torch.from_numpy(rgbt)
    #rgbt = rgbt/255.0

    return rgb.to(device).type(torch.cuda.FloatTensor), rgbt.to(device).type(torch.cuda.FloatTensor)




def eval(dataloader, model):
    for i, (img, ann) in enumerate(dataloader):
        img = img.to(device)
        ann = ann.to(device)
        pred = model(img)
        label_colours = dataloader.dataset.get_pascal_labels()
        label_mask = pred.detach().cpu().numpy()
        print(label_mask)
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        utils.save_image(torch.cat([img, pred, ann], dim=1), 'results.jpeg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch Implementation of Image Segmentation')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = VOC_Dataset()
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle=True, pin_memory=True)
    label_colours = dataset.get_pascal_labels()

    model = Segmentation_Model().to(device)
    optim = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    for epoch in range(args.epochs):
        train(epoch, dataloader, model)