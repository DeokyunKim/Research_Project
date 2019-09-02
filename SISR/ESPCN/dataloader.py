import torch.utils.data as data
from torchvision import transforms
from PIL import Image
from os import listdir
from os.path import join

def is_image_file(filename):
    """Check image file format.
    """
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg'])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

class Dataset(data.Dataset):
    """DIV2K Dataset
    Parameter:
        path (str)      -- DIV2K image data path
        mode (str)      -- train set or validation set 'train' | 'val' 
        upscale_factor (int)  -- upscale_factor 2 | 4 | 8
        We randomly cropped target images and then bilinearly resize the images as network input.
    """
    def __init__(self, path=None, mode = 'train', upscale_factor=4):
        super(Dataset, self).__init__()
        if mode == 'train':
            self.mode = 'train'
        else:
            self.mode = 'valid'
        #get img path

        target_path = join(path, 'DIV2K_{}_HR'.format(self.mode))
        self.target_image_filenames = [join(target_path, x) for x in listdir(target_path) if is_image_file(x)]
        self.totensor = transforms.ToTensor()
        self.transform_target = transforms.Compose([
                                transforms.RandomCrop((128, 128)),
                                ])
        self.upscale_factor = upscale_factor
        resize_ = int(128/self.upscale_factor)
        self.transform_input = transforms.Compose([
                                transforms.Resize((resize_, resize_)),
                                ])
    def __getitem__(self, index):
        image = load_img(self.target_image_filenames[index])
        target_image = self.transform_target(image)
        input_image = self.transform_input(target_image)
        input_image = self.totensor(input_image)
        target_image = self.totensor(target_image)
        return input_image, target_image

    def __len__(self,):
        return len(self.target_image_filenames)

if __name__=='__main__':
    dataset = Dataset(path='./ec1m_landmark_images', mode = 'train', upscale_factor=4)