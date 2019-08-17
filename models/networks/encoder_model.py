"""
Encoder以原始图像作为输入，经过六层3*3卷积，向下采样，InstanceNorm，LReLU后，
做一个Rehape，把他变为(batisize, 1, 1, 8192)的Tensor，然后分别做两个Linear(8196,256)
的全连接得到均值向量和方差向量。
"""

import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 3*3-⬇2-Conv-64, IN, LReLU
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(True),
        )
        # 3*3-⬇2-Conv-128, IN, LReLU
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(True),
        )
        # 3*3-⬇2-Conv-256, IN, LReLU
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(True),
        )
        # 3*3-⬇2-Conv-512, IN, LReLU
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(True),
        )
        # 3*3-⬇2-Conv-64, IN, LReLU
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(True),
        )
        # 3*3-⬇2-Conv-64, IN, LReLU
        self.layer6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(True),
        )
        # 得到均值μ的全连接层
        self.classifier1 = nn.Linear(8192, 256)
        # 得到方差σ^2的全连接层
        self.classifier2 = nn.Linear(8192, 256)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(x.size()[0], -1)
        mean = self.classifier1(x)
        variance = self.classifier2(x)
        return mean, variance

