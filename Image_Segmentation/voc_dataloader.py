import torch
from torch.utils.data import dataset
from os.path import join, splitext
from os import listdir
from PIL import Image
from torchvision import transforms
import numpy as np
from torchvision import utils


class VOC_Dataset(dataset.Dataset):
    def __init__(self, data_path = './dataset/', state= 'train'):
        main_path = join(data_path, 'VOCdevkit', 'VOC2012')
        self.img_path = join(main_path, 'JPEGImages')
        self.ann_path = join(main_path, 'SegmentationClass')

        self.img_list = listdir(self.img_path)
        self.ann_list = listdir(self.ann_path)

        self.img_list.sort()
        self.ann_list.sort()

        self.transform = transforms.Compose([
                                transforms.Resize((256, 256)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ])
        self.n_classes = 21


    def __getitem__(self, index):
        img = Image.open(join(self.img_path, splitext(self.ann_list[index])[0]+'.jpg'))
        ann = Image.open(join(self.ann_path, self.ann_list[index])).convert('RGB').resize((256, 256))

        img = self.transform(img)
        ann = self.encode_segmap(np.array(ann))
        #ann = self.decode_segmap(ann)
        ann = torch.from_numpy(ann) #shape image WxHx(C=1)
        #self.visualize(ann)
        #print(ann)
        return img, ann

    def __len__(self):
        return len(self.ann_list)

    def get_pascal_labels(self, ):
        return np.asarray([
                            [0, 0, 0],
                            [128, 0, 0],
                            [0, 128, 0],
                            [128, 128, 0],
                            [0, 0, 128],
                            [128, 0, 128],
                            [0, 128, 128],
                            [128, 128, 128],
                            [64, 0, 0],
                            [192, 0, 0],
                            [64, 128, 0],
                            [192, 128, 0],
                            [64, 0, 128],
                            [192, 0, 128],
                            [64, 128, 128],
                            [192, 128, 128],
                            [0, 64, 0],
                            [128, 64, 0],
                            [0, 192, 0],
                            [128, 192, 0],
                            [0, 64, 128],
                ])

    def visualize(self, rgb):
        rgb = rgb.permute(2, 0, 1)
        utils.save_image(rgb, 'results.jpeg')

    def encode_segmap(self, mask):
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    def decode_segmap(self, label_mask):
        label_colours = self.get_pascal_labels()
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

        return rgb

if __name__ == '__main__':
    dataset = VOC_Dataset()
    dataset[1]


