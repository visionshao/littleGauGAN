import torch.nn as nn
import models.networks.spade_model as spade
import torch.nn.functional as F


class SpadeResBlk(nn.Module):
    # 来自上一层的图的通道数，输出通道数k，语义图的通道数为label_nc
    def __init__(self, norm_nc, k, label_nc):
        super().__init__()

        # 两层SPADE

        self.norm0 = spade.Spade(norm_nc, label_nc)
        self.norm1 = spade.Spade(k, label_nc)

        self.conv_0 = nn.Conv2d(norm_nc, k, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(k, k, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        dx = self.conv_0(self.actvn(self.norm0(x, segmap)))
        dx = self.conv_1(self.actvn(self.norm1(dx, segmap)))

        out = dx

        return out

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
