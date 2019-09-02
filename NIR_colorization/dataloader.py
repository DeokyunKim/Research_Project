import torch.nn
import torch.utils.data as data
from os.path import join
from os import listdir
from torchvision import transforms
import torchvision.transforms.functional as F
from random import *
from skimage import io, color
import numpy as np

class RGB_NIR_Dataset(data.Dataset):
    def __init__(self, path='./dataset/nirscene1', mode='train'):
        #train, val, test, 
        self.mode = mode
        img_path = listdir(path)
        img_paths = list()
        for path_ in img_path:
            path_ = join(path, path_)
            img_paths.extend([join(path_, x) for x in listdir(path_)])
        self.rgb_images = list()
        self.nir_images = list()
        for path in img_paths:
            if 'rgb' in path:
                self.rgb_images.append(path)
            else:
                self.nir_images.append(path)
        self.rgb_images.sort()
        self.nir_images.sort()

    def __getitem__(self, index):
        rgb_image = io.imread(self.rgb_images[index])
        nir_image = io.imread(self.nir_images[index])
        h, w, c = rgb_image.shape
        cropped_x, cropped_y = 256, 256
        x,y = randint(0, w-cropped_x), randint(0, h-cropped_y)

        rgb_image = rgb_image[y:y+cropped_y, x:x+cropped_x]
        nir_image = nir_image[y:y+cropped_y, x:x+cropped_x]
        rgb_image = np.float32(rgb_image)
        nir_image = np.float32(nir_image)

        rgb_image = torch.from_numpy(rgb_image).float().permute(2,0,1)
        rgb_image = rgb_image.div(255)
        nir_image = torch.from_numpy(nir_image[:,:,np.newaxis]).float().permute(2,0,1)
        nir_image = nir_image.div(250)

        return rgb_image, nir_image

    def __len__(self,):
        return len(self.rgb_images)
