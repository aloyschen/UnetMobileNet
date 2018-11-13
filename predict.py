import cv2
import config
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from MobileUNet import MobileUNet
from torchvision import transforms

def predict_image(image_file, model_path):
    """
    Introduction
    ------------
        使用MobileNet-UNet预测图像
    """
    image = cv2.imread(image_file)
    image = cv2.resize(image, (config.image_size, config.image_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)
    model = MobileUNet()
    ckpt = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(ckpt)
    model.eval()
    output_mask = model(image_tensor)
    output_mask = output_mask.reshape(config.image_size // 2, config.image_size // 2)
    image_mask = output_mask.detach().numpy() * 255
    image_mask = image_mask.astype(np.uint8)
    image_mask = cv2.resize(image_mask, (config.image_size, config.image_size))
    plt.subplot(121)
    plt.imshow(image)
    plt.subplot(122)
    plt.imshow(image_mask)
    plt.show()



if __name__ == '__main__':
    predict_image('./test1.jpg', './train_model_epoch51.pth')
