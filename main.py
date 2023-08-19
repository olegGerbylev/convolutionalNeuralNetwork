from io import BytesIO

import numpy as np
import requests

import torch
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F

import matplotlib.pyplot as plt
import cv2
import os

from PIL import Image
from tqdm import tqdm


# ds_mnist = tv.datasets.MNIST('./datasets', download=True, transform=tv.transforms.Compose([
#     tv.transforms.ToTensor()
# ]))
#
# ds_mnist[0][0].numpy()[0].shape
# # len(ds_mnist)
#
# batch_size = 16
#
# dataloader = torch.utils.data.DataLoader(ds_mnist, batch_size, shuffle=True, num_workers=1, drop_last=True)
#
# for img, label in dataloader:
#     print(img.shape)
#     print(label.shape)
#     break
#
#
# class Neural_numbers(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flat = nn.Flatten()
#         self.linear1 = nn.Linear(28 * 28, 100)
#         self.linear2 = nn.Linear(100, 10)
#         self.act = nn.ReLU()
#
#     def forward(self, x):
#         out = self.flat(x)
#         out = self.linear1(out)
#         out = self.act(out)
#         out = self.linear2(out)
#         return out
#
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
# model = Neural_numbers()
#
# print(count_parameters(model))
#
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
#
# def accuracy(pred, label):
#     answer = torch.nn.functional.softmax(pred.detach()).numpy().argmax(1) == label.numpy().argmax(1)
#     return answer.mean()
#
# epochs = 10
# for epoch in range(epochs):
#     loss_val = 0
#     acc_val = 0
#     for img, label in (pbar:= tqdm(dataloader)):
#         optimizer.zero_grad()
#
#         label = nn.functional.one_hot(label, 10).float()
#         pred = model(img)
#
#         loss = loss_fn(pred, label)
#         loss.backward()
#         loss_val += loss.item()
#         optimizer.step()
#
#         acc_current = accuracy(pred, label)
#         acc_val += acc_current
#         pbar.set_description(f'loss:{loss.item():.4f}\t accuracy:{acc_current:.3f}')
#     print( loss_val/len(dataloader))
#     print(acc_val / len(dataloader))
#
# print(accuracy(pred, label))


class Dataset2class(torch.utils.data.Dataset):
    def __init__(self, dir1, dir2):
        super().__init__()
        self.path_dir1 = dir1
        self.path_dir2 = dir2
        self.dir1_list = sorted(os.listdir(self.path_dir1))
        self.dir2_list = sorted(os.listdir(self.path_dir2))

    def __getitem__(self, item):
        if item < len(self.dir1_list):
            class_id = 0
            img_path = os.path.join(self.path_dir1, self.dir1_list[item])
        else:
            class_id = 1
            item -= len(self.dir1_list)
            img_path = os.path.join(self.path_dir2, self.dir2_list[item])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img / 255.0
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        img = img.transpose((2, 0, 1))
        t_img = torch.from_numpy(img)
        t_class_id = torch.tensor(class_id)

        return {'img':t_img, 'label': t_class_id}

    def __len__(self):
        return len(self.dir1_list) + len(self.dir2_list)


train_dogs_path = './training_set/training_set/dogs'
train_cats_path = './training_set/training_set/cats'
test_dogs_path = './test_set/test_set/dogs'
test_cats_path = './test_set/test_set/cats'

train_ds_catsdogs = Dataset2class(
    train_dogs_path,
    train_cats_path
)
test_ds_catsdogs = Dataset2class(
    test_dogs_path,
    test_cats_path
)




batch_size = 16

train_loader = torch.utils.data.DataLoader(train_ds_catsdogs, batch_size=batch_size, num_workers=1, shuffle=True, drop_last= True)
test_loader = torch.utils.data.DataLoader(test_ds_catsdogs, batch_size=batch_size, num_workers=1, shuffle=True, drop_last= False)

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv0 = nn.Conv2d(3, 32, 3, 1, 0)
        self.conv1 = nn.Conv2d(32, 32, 3, 1, 0)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 0)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 0)
        self.adaptivepool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64,10)
        self.linear2 = nn.Linear(10,2)

    def forward(self, x):

        out = self.conv0(x)
        out = self.act(out)
        out = self.maxpool(out)

        out = self.conv1(out)
        out = self.act(out)
        out = self.maxpool(out)

        out = self.conv2(out)
        out = self.act(out)
        out = self.maxpool(out)

        out = self.conv3(out)
        out = self.act(out)

        out = self.adaptivepool(out)
        out = self.flatten(out)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)


        return out

net = ConvNet()
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999))

def accuracy(pred, label):
    answer = torch.nn.functional.softmax(pred.detach()).numpy().argmax(1) == label.numpy().argmax(1)
    return answer.mean()

epochs = 10
for epoch in range(epochs):
    loss_val = 0
    acc_val = 0
    for sample in (pbar:= tqdm(train_loader)):
        img = sample['img']
        label = sample['label']
        optimizer.zero_grad()

        label = nn.functional.one_hot(label, 2).float()
        pred = net(img)

        loss = loss_fn(pred, label)
        loss.backward()
        loss_val += loss.item()
        optimizer.step()

        acc_current = accuracy(pred, label)
        acc_val += acc_current
        pbar.set_description(f'loss:{loss.item():.4f}\t accuracy:{acc_current:.3f}')
    print( loss_val/len(train_loader))
    print(acc_val / len(train_loader))

print(accuracy(pred, label))

print("fdsfsfdsdfsddfs")

loss_val = 0
acc_val = 0
for sample in (pbar:= tqdm(test_loader)):
    with torch.no_grad():
        img = sample['img']
        label = sample['label']

        label = nn.functional.one_hot(label, 2).float()
        pred = net(img)

        loss = loss_fn(pred, label)
        loss_val += loss.item()

        acc_current = accuracy(pred, label)
        acc_val += acc_current
    pbar.set_description(f'loss:{loss.item():.4f}\t accuracy:{acc_current:.3f}')
print( loss_val/len(train_loader))
print(acc_val / len(train_loader))