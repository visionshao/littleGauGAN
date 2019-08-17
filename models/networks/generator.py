import torch
import torch.nn as nn
import torch.nn.functional as F
import models.networks.spaderesblk_model as SPADERB


class GANGenerator(nn.Module):
    def __init__(self):

        super(GANGenerator, self).__init__()
        # 在这里固定语义图的Chanels为3
        label_nc = 3

        # 定义SPADEGenerator各层
        self.lin = nn.Linear(256, 16384)

        self.head_0 = SPADERB.SpadeResBlk(1024, 1024, label_nc)
        self.G_middle_0 = SPADERB.SpadeResBlk(1024, 1024, label_nc)
        self.G_middle_1 = SPADERB.SpadeResBlk(1024, 1024, label_nc)

        self.up_0 = SPADERB.SpadeResBlk(1024, 512, label_nc)
        self.up_1 = SPADERB.SpadeResBlk(512, 256, label_nc)
        self.up_2 = SPADERB.SpadeResBlk(256, 128, label_nc)
        self.up_3 = SPADERB.SpadeResBlk(128, 64, label_nc)

        self.conv3 = nn.Conv2d(64, 3, stride=1, kernel_size=3, padding=1)

        #上采样
        self.up = nn.Upsample(scale_factor=2)

    # x为以Encoder输出的均值和方差作为均值方差的正态分布向量，segmap是语义图（256*256）
    def forward(self, x, segmap):
        x = x.view(x.size()[0], -1)
        x = self.lin(x)
        x = x.view(x.size()[0], 1024, 4, 4)  # 图像大小为4*4， 通道数为1024，注意pytorch卷积的输入通道数在高，宽前。
        x = self.head_0(x, segmap)
        x = self.up(x)  # 图像大小为8 * 8， 通道数1024
        x = self.G_middle_0(x, segmap)
        x = self.up(x)  # 图像大小为16 * 16， 通道数1024
        x = self.G_middle_1(x, segmap)
        x = self.up(x)  # 图像大小为32 * 32， 通道数1024
        x = self.up_0(x, segmap)
        x = self.up(x)  # 图像大小为64 * 64， 通道数512
        x = self.up_1(x, segmap)
        x = self.up(x)  # 图像大小为128 * 128， 通道数256
        x = self.up_2(x, segmap)
        x = self.up(x)  # 图像大小为256 * 256， 通道数128
        x = self.up_3(x, segmap)   # 图像大小为256 * 256， 通道数64
        x = self.conv3(x)  # 图像大小为256 * 256， 通道数3
        output = F.tanh(x)

        return output



