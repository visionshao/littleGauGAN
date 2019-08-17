import models.networks.discriminator as D_model
import models.networks.generator as G_model
import models.networks.encoder_model as E_model
import torch
import torch.nn as nn



class GauGAN(nn.Module):

    def __init__(self):

        super(GauGAN, self).__init__()
        self.G = G_model.GANGenerator()
        self.D = D_model.Discriminator()
        self.E = E_model.Encoder()

    def forward(self, semantic_img, real_img=None):
        # 如果是在训练
        if real_img is not None:
            mean, var = self.E.forward(real_img)
            # print(mean.size())  # batch * 256
            # print(var.size())   # batch * 256
            z = self.reparameterize(mean, var)

            G_out = self.G.forward(x=z, segmap=semantic_img)    # batch * 3 * 256 * 256

            D_real_input = torch.cat((real_img, semantic_img), dim=1)
            D_real_out = self.D.forward(D_real_input)

            D_fake_input = torch.cat((G_out, semantic_img), dim=1)
            D_fake_out = self.D.forward(D_fake_input)

            return D_real_out, D_fake_out, G_out
        else:
            z = torch.randn(size=[semantic_img.size()[0], 256])
            G_out = self.G.forward(x=z, segmap=semantic_img)
            # print(G_out.size())

            return G_out

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.__mul__(std) + mu




