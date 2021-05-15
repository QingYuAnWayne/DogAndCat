import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from math import ceil


class DogCat(data.Dataset):
    def __init__(self, root, transforms=None, train=True, test=False):
        """
        目标：获取所有图片地址，并根据训练、验证、测试划分数据
        """
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(path=root)]

        # test1: data/test1/8973.jpg
        # train: data/train/cat.10004.jpg
        def get_index(x):
            if not self.test:
                x = x.split('.')[-2]
            else:
                x = x.split('.')[-2].split('/')[-1]
            return int(x)

        imgs = sorted(imgs, key=lambda x: get_index(x))
        imgs_num = len(imgs)

        # 划分训练、验证集，验证:训练 = 3:7
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:ceil(0.7 * imgs_num)]
        else:
            self.imgs = imgs[ceil(0.7 * imgs_num):]

        if transforms is None:
            # 数据转换操作，测试验证和训练的数据转换有所区别
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            # 测试集和验证集
            if self.test or not train:
                self.transform = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transform = T.Compose([
                    T.Resize(256),
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        """
        返回一张图片的数据
        对于测试集，没有label，返回图片id，如1000.jpg返回1000
        """
        img_path = self.imgs[index]
        if self.test:
            label = int(img_path.split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_path.split('.')[-3].split('/') else 0
        data = Image.open(img_path)
        data = self.transform(data)
        return data, label

    def __len__(self):
        """
        返回数据集中所有图片的个数
        """
        return len(self.imgs)
