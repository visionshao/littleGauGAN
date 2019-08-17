import torch.nn as nn
import torch.nn.functional as F


class Spade(nn.Module):
    # generate_image和label_image的通道数
    def __init__(self, norm_nc, label_nc):

        super().__init__()
        # SPADE固定的通道数128，处理语义图
        nhidden = 128
        # normalize generate_image
        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        # label_image的第一层卷积
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        # label_image的gamma卷积
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        # label_image的beta卷积
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        # 标准化
        normalized = self.param_free_norm(x)
        # resize语义图以使之与生成的输入图大小相同
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        out = normalized * (1 + gamma) + beta

        return out




