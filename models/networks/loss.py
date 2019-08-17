import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.vgg import VGG19

"""
设计GAN的损失函数
"""


class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.Tensor = tensor

    # 得到相应标签，和输入的大小相同
    def get_target_tensor(self, input, target_is_real):
        # 如果目标值为真（真实图像），得到与input相同size的1标签
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        # 如果目标值为假（生成图像），得到与input相同size的0标签
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def loss(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        loss = F.binary_cross_entropy_with_logits(input=input, target=target_tensor)
        return loss

    def __call__(self, input, target_is_real):
        return self.loss(input, target_is_real)


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

