import os
import torch
import config
from torch import optim
from MobileUNet import MobileUNet
from dataset import LFWHairDataset
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate
from torchvision.transforms import Compose, ColorJitter, RandomResizedCrop, RandomAffine, RandomRotation, RandomHorizontalFlip, ToTensor

# 指定使用GPU的Index
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_index


def dice_loss(scale=None):
    def fn(input, target):
        smooth = 1.

        if scale is not None:
            scaled = interpolate(input, scale_factor = scale, mode = 'bilinear', align_corners = False)
            iflat = scaled.view(-1)
        else:
            iflat = input.view(-1)

        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

    return fn

def train():
    """
    Introduction
    ------------
        训练图像分割模型
    """
    print('Start bulid dataset')
    device = torch.device("cuda" if config.cuda else "cpu")
    train_transform = Compose([
        ColorJitter(0.3, 0.3, 0.3, 0.3),
        RandomResizedCrop(config.image_size, scale=(0.8, 1.2)),
        RandomAffine(10.),
        RandomRotation(13.),
        RandomHorizontalFlip(),
        ToTensor(),
    ])
    train_dataset = LFWHairDataset(config.image_dir, config.mask_dir, config.image_size, training = True, transform = train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size = config.train_batch, shuffle = True, num_workers = 2, pin_memory = True)
    print('train on {} samples'.format(train_dataset.samples_num))
    model = MobileUNet(class_num = config.class_num, pre_train = config.MobileNetV2_weights)
    if config.cuda is True:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr = config.learning_rate)
    criterion = dice_loss(scale = 2)
    for epoch in range(config.Epoch_num):
        model.train()
        for index, (image, mask) in enumerate(train_dataloader):
            image = image.to(device)
            mask = mask.to(device)
            mask_pred = model(image)
            loss = criterion(mask_pred, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if index % config.print_feq == 0:
                print("Epoch: {} Batch: {}/{} loss: {:.4f}".format(epoch, index, len(train_dataloader), loss.item()))
        if epoch % config.save_feq == 0:
            print('save model')
            torch.save(model.state_dict(), config.model_dir + 'train_model_epoch{}.pth'.format(epoch + 1))


if __name__ == '__main__':
    train()
