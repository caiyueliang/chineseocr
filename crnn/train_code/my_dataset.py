#!/usr/bin/python
# encoding: utf-8

import os
import random
import torch
import base64
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import numpy as np
import cv2
from io import BytesIO


class MyDataset(Dataset):
    def __init__(self, root=None, transform=None, target_transform=None, is_train=False, img_h=110, img_w=32, nc=3):
        self.root = root
        self.sample_list = list()
        self.is_train = is_train
        self.img_h = img_h
        self.img_w = img_w
        self.nc = nc

        for root, dirs, files in os.walk(self.root):
            for file in files:
                self.sample_list.append(os.path.join(root, file))
        # with open(self.label_file, 'r') as f:
        #     self.sample_list = f.readlines()

        self.nSamples = len(self.sample_list)
        # print(self.sample_list)
        print('label_file: root: %s samples len: %d' % (self.root, self.nSamples))

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    # PIL
    def __getitem__(self, index):
        assert index < self.nSamples, 'index range error'
        image_path = self.sample_list[index]
        label = self.sample_list[index].replace(' ', '').split('_')[-1].split('.')[0]
        # record = self.sample_list[index].replace('\n', '').replace('\r', '').replace(' ', '')
        # str_list = record.split(':')
        # label = str_list[-1]
        # image_path = str_list[0]

        # print("image_path", image_path)
        # print("label", label)
        img = Image.open(image_path)
        # if self.is_train is True:
        img = self.random_base64(img)               # 随机转换成base64再转回来

        if self.nc == 1:                            # 通道数为1，图片转灰度图
            img = img.convert('L')                  # 转灰度图

        # img_1 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        # cv2.imshow("old", img_1)

        if self.is_train is True:
            img = self.random_gaussian(img)
            img = self.random_bright(img)
            img = self.random_crop(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        # print(img)
        # print(img.size())
        # print(label)
        return (img, label)

    # ============================================================================================
    def pil_base64(self, image):
        img_buffer = BytesIO()
        image.save(img_buffer, format='jpg')
        byte_data = img_buffer.getvalue()
        base64_str = base64.b64encode(byte_data)
        return base64_str

    def base64_pil(self, base64_str):
        image = base64.b64decode(base64_str)
        image = BytesIO(image)
        image = Image.open(image)
        return image

    def cv2_to_base64(self, image):
        base64_str = cv2.imencode('.jpg', image)[1].tostring()
        base64_str = base64.b64encode(base64_str)
        return base64_str

    def base64_to_cv2(self, base64_str):
        imgString = base64.b64decode(base64_str)
        nparr = np.fromstring(imgString, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image

    def random_base64(self, image):
        k = random.random()
        if k > 0.5:
            image_base64 = self.pil_base64(image)
            image = self.base64_pil(image_base64)
        return image

    # ============================================================================================
    # 随机高斯模糊(PIL)
    def random_gaussian(self, img, max_n=2):
        # img_1 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        # cv2.imshow("old", img_1)

        k = random.random()
        if k > 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=1.3))
            # img = cv2.GaussianBlur(img, ksize=(k, k), sigmaX=1.5)

        # img_2 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        # cv2.imshow("new", img_2)
        # cv2.waitKey()
        return img

    # 随机裁剪(PIL)
    def random_crop(self, img, max_n=4):
        imw, imh = img.size
        # img_1 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        # cv2.imshow("old", img_1)

        top = random.randint(0, max_n)
        bottom = random.randint(0, max_n)
        left = random.randint(0, max_n)
        right = random.randint(0, 2)
        # print top, bottom, left, right

        # (left, upper, right, lower)
        box = (left, top, imw-right, imh-bottom)
        roi = img.crop(box)
        roi = roi.resize((imw, imh))

        # print(roi.size)
        roi = roi.resize((imw, imh))
        # print(roi.size)
        # img_2 = cv2.cvtColor(np.asarray(roi), cv2.COLOR_RGB2BGR)
        # cv2.imshow("new", img_2)
        # cv2.waitKey()

        return roi

    # 随机调亮(opencv)
    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.6:
            if self.nc == 1:  # 通道数为1，图片转灰度图
                im = cv2.cvtColor(np.asarray(im), cv2.COLOR_GRAY2BGR)
            else:
                im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
            im = im * alpha + random.randrange(-delta, delta)
            im = im.clip(min=0, max=255).astype(np.uint8)
            if self.nc == 1:  # 通道数为1，图片转灰度图
                im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
            else:
                im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        return im

    # # opencv
    # def __getitem__(self, index):
    #     assert index < self.nSamples, 'index range error'
    #     record = self.sample_list[index].replace('\n', '').replace('\r', '').replace(' ', '')
    #     str_list = record.split(':')
    #     label = str_list[-1]
    #     image_path = str_list[0]
    #
    #     # img = Image.open(os.path.join(self.root, image_path))
    #     img = cv2.imread(os.path.join(self.root, image_path))
    #
    #     if self.is_train is True:
    #         img = self.random_gaussian(img)
    #         img = self.random_crop(img)
    #
    #     img = cv2.resize(img, (self.img_w, self.img_h))
    #     if self.transform is not None:
    #         img = self.transform(img)
    #
    #     if self.target_transform is not None:
    #         label = self.target_transform(label)
    #
    #     # print(img)
    #     # print(img.size())
    #     # print(label)
    #     return (img, label)
    #
    # # 随机高斯模糊(opencv)
    # def random_gaussian(self, img, max_n=3):
    #     # img_1 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    #     # cv2.imshow("old", img_1)
    #
    #     k = random.random()
    #     if k > 0.5:
    #         img = cv2.GaussianBlur(img, ksize=(max_n, max_n), sigmaX=1.5)
    #
    #     # img_2 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    #     # cv2.imshow("new", img_2)
    #     # cv2.waitKey()
    #     return img
    #
    # # 随机裁剪(opencv)
    # def random_crop(self, img, max_n=5):
    #     imh, imw, _ = img.shape
    #     # img_1 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    #     # cv2.imshow("old", img_1)
    #
    #     top = random.randint(0, max_n)
    #     bottom = random.randint(0, max_n)
    #     left = random.randint(0, max_n)
    #     right = random.randint(0, max_n)
    #     # print top, bottom, left, right
    #
    #     # (left, upper, right, lower)
    #     corp_img = img[top:imh-bottom, left:imw-right]
    #     # box = (left, top, imw-right, imh-bottom)
    #     # roi = img.crop(box)
    #     roi = cv2.resize(corp_img, (imw, imh))
    #
    #     # print(roi.size)
    #     # roi = roi.resize((imw, imh))
    #     # print(roi.size)
    #     # img_2 = cv2.cvtColor(np.asarray(roi), cv2.COLOR_RGB2BGR)
    #     # cv2.imshow("new", img_2)
    #     # cv2.waitKey()
    #
    #     return roi


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels
