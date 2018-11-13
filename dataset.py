import os
from PIL import Image
import torch.utils.data as data

class LFWHairDataset(data.Dataset):
    def __init__(self, image_dir, mask_dir, image_size, training, transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.training = training
        self.transform = transform
        self.image_size = image_size
        self.mask_files = []
        self.image_files = []
        for file in os.listdir(mask_dir):
            self.mask_files.append(os.path.join(self.mask_dir, file))
            image_subdir = file.split(".")[0].split('_')
            if len(image_subdir) > 1:
                image_file = os.path.join(self.image_dir, '_'.join(image_subdir[: -1]), file.split('.')[0] + '.jpg')
                self.image_files.append(image_file)
        self.samples_num = len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        mask_file = self.mask_files[index]
        image = Image.open(image_file)
        mask = Image.open(mask_file)
        # 只获取头发的通道
        mask = mask.split()[0]

        if self.training == True:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask


    def __len__(self):
        return self.samples_num
