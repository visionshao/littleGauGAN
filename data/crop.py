import os
from torch.utils import data
from PIL import Image


class Cropper(data.Dataset):
    def __init__(self, img_root, label_root):
        # 所有图片的相对路径
        imgs = os.listdir(img_root)
        labels = os.listdir(label_root)
        # 所有图片的绝对路径
        self.imgs = [os.path.join(img_root, k) for k in imgs]
        self.labels = [os.path.join(label_root, k) for k in labels]

        for img_path, label_path in zip(self.imgs, self.labels):
            pil_img = Image.open(img_path)
            pil_img = pil_img.resize((256, 256), Image.ANTIALIAS)

            pil_label = Image.open(label_path)
            pil_label = pil_label.resize((256, 256), Image.ANTIALIAS)
            # 保存到原来的位置
            pil_img.save(img_path)
            pil_label.save(label_path)
