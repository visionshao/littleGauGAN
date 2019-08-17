import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
])


# 定义自己的数据集合
class FlameSet(data.Dataset):
    def __init__(self, img_root, label_root):
        # 所有图片的绝对路径
        imgs = os.listdir(img_root)
        labels = os.listdir(label_root)
        self.imgs = [os.path.join(img_root, k) for k in imgs]
        self.labels = [os.path.join(label_root, k) for k in labels]
        # 归一化处理
        self.transforms = transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label_path = self.imgs[index]
        pil_img = Image.open(img_path)
        pil_label = Image.open(label_path)
        if self.transforms:
            img = self.transforms(pil_img)
            label = self.transforms(pil_label)
        else:
            pil_img = np.asarray(pil_img)
            pil_label = np.asarray(pil_label)
            img = torch.from_numpy(pil_img)
            label = torch.from_numpy(pil_label)
        return img, label

    def __len__(self):
        return len(self.imgs)



