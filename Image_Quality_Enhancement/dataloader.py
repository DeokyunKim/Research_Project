
import torch.utils.data as data
from PIL import Image
from PIL import ImageEnhance
import os
from torchvision import transforms 
import random
import numpy as np

class Fivek_Dataset(data.Dataset):
    def __init__(self, main_path='./dataset/', mode='train'):
        self.mode = mode
        self.input_path = os.path.join(main_path, 'FiveK_Lightroom_Export_InputDayLight')
        self.target_path = os.path.join(main_path,'expertC')

        self.list_input_path = os.listdir(self.input_path)
        self.list_target_path = os.listdir(self.target_path)

        self.list_input_path.sort()
        self.list_target_path.sort()

        #TODO image crop & convert Tensor
        self.transforms = transforms.ToTensor()
        if self.mode=='train':
            self.start = 0
        elif self.mode=='val':
            self.start = 3500
        elif self.mode=='test':
            self.start = 4500
        else:
            raise Exception('The mode [%s] is not define'%self.mode)

    def __getitem__(self, index):
        input_image = Image.open(os.path.join(self.input_path, self.list_input_path[self.start+index]))
        target_image = Image.open(os.path.join(self.target_path, self.list_input_path[self.start+index])) #input and target image file's name is same

        if self.mode == 'test':
            #width, height, channel = input_image.shape
            #input_image = input_image.resize((512, 512))
            #target_image = target_image.resize((512, 512))
            #resized_width, resized_height = 512, 512
            #resized_input_image = input_image.resize((resized_width, resized_height))
            #resized_target_image = target_image.resize((resized_width, resized_height))
            return self.transforms(input_image), self.transforms(target_image)

        elif self.mode == 'val':
            resized_width, resized_height = 512, 512
            resized_input_image = input_image.resize((resized_width, resized_height))
            resized_target_image = target_image.resize((resized_width, resized_height))
            resized_input_image = self.transforms(resized_input_image)
            resized_target_image = self.transforms(resized_target_image)
            return resized_input_image, resized_target_image

        else:
            width, height = input_image.size #input and target size are same.
            target_w, target_h = target_image.size
            cropping_width, cropping_height = 512, 512
            x,y = random.randint(0, width-cropping_width), random.randint(0, height-cropping_height)
            cropping_area = (x, y, x+cropping_width, y+cropping_height)
            cropped_input_image = input_image.crop(cropping_area)
            cropped_target_image = target_image.crop(cropping_area)

            #vflip
            if random.uniform(0, 1) > 0.5:
                cropped_input_image = cropped_input_image.transpose(Image.FLIP_TOP_BOTTOM)
                cropped_target_image = cropped_target_image.transpose(Image.FLIP_TOP_BOTTOM)

            #hflip
            if random.uniform(0, 1) > 0.5:
                cropped_input_image = cropped_input_image.transpose(Image.FLIP_LEFT_RIGHT)
                cropped_target_image = cropped_target_image.transpose(Image.FLIP_LEFT_RIGHT)

            #brightness
            if random.uniform(0, 1) > 0.5:
                input_enhancer = ImageEnhance.Brightness(cropped_input_image)
                cropped_input_image = input_enhancer.enhance(random.uniform(1.5, 2.0)) 

            #gaussian noise
            #if random.uniform(0, 1) > 0.5:
            """
            if True:
                mean = 0
                var = 10
                sigma = var ** 0.5
                gaussian = np.random.normal(mean, sigma, (cropping_width, cropping_height, 3))
                cropped_input_image = np.asarray(cropped_input_image)
                cropped_input_image = Image.fromarray(np.uint8(cropped_input_image + gaussian))
            """
            cropped_input_image = self.transforms(cropped_input_image)
            cropped_target_image = self.transforms(cropped_target_image)

            return cropped_input_image, cropped_target_image
        
    def __len__(self,):
        if self.mode=='train':
            return 3500
        elif self.mode=='val':
            return 1000
        elif self.mode=='test':
            return 500
        else:
            raise Exception('The mode [%s] is not define'%self.mode)

if __name__ == '__main__':
    dataset = Fivek_Dataset(mode='val')
    dataset[0][0].save('input_sample.jpeg')
    #for i in range(len(dataset)):
    #    dataset[i]
        #print(i)
    
    
