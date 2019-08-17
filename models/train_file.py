from models.main_model import *
import torch
import tqdm
import data.dataloader as dataload
from torchvision import transforms
from torch.utils.data import DataLoader
from data.crop import *
import models.networks.loss as loss_model

# 模型保存路径，训练集中真实图路径，训练集中语义图路径。
model_path = r'D:\pytorch_codes\littleGauGAN\trained_models\littleGan.pkl '
Ima_path = r'D:\pytorch_codes\littleGauGAN\datasets\ADK\imgs'
Label_path = r'D:\pytorch_codes\littleGauGAN\datasets\ADK\annotations'

# 预处理（剪切，数据加载器）
Cropper(Ima_path, Label_path)
dataset = dataload.FlameSet(Ima_path, Label_path)
data = DataLoader(dataset, batch_size=4, shuffle=False)

# 用于将G生成的结果（张量）转化为满足RGB格式的图像
# 这一步实际上相当于dataset.FlameSet中的transform相反操作
transform1 = transforms.ToPILImage(mode="RGB")

# 模型及损失函数类实例化
littleGan = GauGAN()
loss_f = loss_model.GANLoss()
v_loss = loss_model.VGGLoss().cuda()

# 训练模式
Train = True

if Train:
    # 训练100000个epoch
    for i in tqdm.trange(100000):
        for real, seman in data:

            D_r, D_f, G_o = littleGan.forward(real, seman)
            # 生成器损失=对抗损失 + 0.5 * 真实图与G的生成图的VGG损失
            # 这里我们额外引入真实图与G的生成图的VGG损失，希望能更好地训练G
            loss_pixel = v_loss(G_o.cuda(), real.cuda())
            loss_GAN = loss_f(D_f, True)

            loss_G = loss_GAN + 0.5 * loss_pixel
            print("Has get loss_G")

            # 判别器损失 = 真实图像损失 + 生成图像损失
            loss_real = loss_f(D_r, True)
            loss_fake = loss_f(D_f, False)
            loss_D = loss_real + loss_fake
            print("Has get loss_D")

            # 优化器
            opt_D = torch.optim.SGD(littleGan.D.parameters(), lr=0.0001, momentum=0.99)
            opt_G = torch.optim.Adam(littleGan.G.parameters(), lr=0.0001)
            opt_E = torch.optim.Adam(littleGan.E.parameters(), lr=0.0001)
            # 训练D
            opt_D.zero_grad()
            loss_D.backward(retain_graph=True)
            opt_D.step()
            print("Has trained D")
            # 训练G
            opt_G.zero_grad()
            loss_G.backward(retain_graph=True)
            opt_G.step()
            print("Has trained G")
            # 训练E
            opt_E.zero_grad()
            loss_pixel.backward(retain_graph=True)
            opt_E.step()
            print("Has trained D")

        # 保存模型
        if i % 50 == 0:
            littleGan.eval()
            torch.save(littleGan.state_dict(), model_path)
# 测试
else:
    # 加载模型
    littleGan.load_state_dict(torch.load(model_path))
    # 一个batch中有4个图像，选择输出并保存第四张图像
    i = 3
    # 这里就直接使用训练集数据了，实际上应该使用测试集数据
    for real, seman in data:
        G_out = littleGan.forward(seman)
        im = transform1(G_out[i])
        im = im.resize((256, 256))
        im.save(r"D:\pytorch_codes\littleGauGAN\out\\" + str(i) + ".jpg")

