import numpy as np
import torch.nn as nn

"""
GAN判别器
"""


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        # real_img是语义图和真实图连接结果，batch * 6 * 256 * 256
        # fake_img是语义图和generator生成图像连接结果，batch * 6 * 256 * 256

        models = [[nn.Conv2d(6, 64, kw, stride=2, padding=padw),  # 64 * 128 * 128
                   nn.InstanceNorm2d(64),
                   nn.LeakyReLU(0.2, False)],
                  [nn.Conv2d(64, 128, kw, stride=2, padding=padw),  # 128 * 64 * 64
                   nn.InstanceNorm2d(128),
                   nn.LeakyReLU(0.2, False),
                   nn.Sigmoid()],
                  [nn.Conv2d(128, 256, kw, stride=2, padding=padw),  # 256 * 32 * 32
                   nn.InstanceNorm2d(256),
                   nn.LeakyReLU(0.2, False),
                   nn.Sigmoid()],
                  [nn.Conv2d(256, 512, kw, stride=1, padding=1),  # 512 * 31 * 31
                   nn.InstanceNorm2d(512),
                   nn.LeakyReLU(0.2, False)],
                  [nn.Conv2d(512, 1, kw, stride=1, padding=1),
                   nn.Sigmoid()]]  # 1 * 30 * 30

        for i in range(len(models)):
            self.add_module("model" + str(i), nn.Sequential(*models[i]))

    def forward(self, x):
        out = [x]
        for submodel in self.children():
            intermediate_output = submodel(out[-1])
            out.append(intermediate_output)
        return out[-1]

# d = Discriminator()
# r = d.forward(torch.zeros([3, 6, 256, 256]))
# for i in r:
#     print(i.size())