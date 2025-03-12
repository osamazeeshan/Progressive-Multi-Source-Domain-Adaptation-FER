import os

import string
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.io import read_image
import pandas as pd

from PIL import Image

import torchvision.transforms as T

import glob

import config


class BSDS500(Dataset):

    def __init__(self):

        image_folder = config.CURRENT_DIR + config.DATASET_FOLDER + '/BSDS300/images/train'

        # self.image_files = img_files
        # temp = list(map(3, '/home/osamazeeshan/Downloads/PhD/FER/code/domain-adaptation-playground/data/BSDS300/images/train/187071.jpg'))
        # print(temp)
        # self.image_files = list(glob.glob('/*.jpg'))

        # get every image in the folder ending with .jpg and add to the image list
        self.image_files = glob.glob(image_folder + '/*.jpg')

    def __getitem__(self, i):
        image = cv2.imread(self.image_files[i], cv2.IMREAD_COLOR)
        tensor = torch.from_numpy(image.transpose(2, 0, 1))
        return tensor

    def __len__(self):
        return len(self.image_files)


class MNISTM(Dataset):

    def __init__(self, train=True):
        super(MNISTM, self).__init__()
        self.mnist = datasets.MNIST(config.CURRENT_DIR + config.DATASET_FOLDER + '/mnist', train=train, download=True)
        self.bsds = BSDS500()

        # Fix RNG so the same images are used for blending
        self.rng = np.random.RandomState(42)

    def __getitem__(self, i):
        digit, label = self.mnist[i]
        digit = transforms.ToTensor()(digit)
        bsds_image = self._random_bsds_image()
        patch = self._random_patch(bsds_image)
        patch = patch.float() / 255
        blend = torch.abs(patch - digit)

        # print(blend)
        # print(label)
        return blend, label

    def _random_patch(self, image, size=(28, 28)):
        _, im_height, im_width = image.shape
        x = self.rng.randint(0, im_width-size[1])
        y = self.rng.randint(0, im_height-size[0])
        return image[:, y:y+size[0], x:x+size[1]]

    def _random_bsds_image(self):
        i = self.rng.choice(len(self.bsds))
        return self.bsds[i]

    def __len__(self):
        return len(self.mnist)   


class FerDatasets(Dataset):

    def __init__(self, imgs, labels, flag = 0):
        super(FerDatasets, self).__init__()

        self.img = imgs
        self.label = labels
        self.flag = flag

    def __getitem__(self, i):
        # img = read_image(self.img[i])

        # img = transforms.Resize(100)(img)

        img = cv2.imread(self.img[i], cv2.IMREAD_COLOR)
        img1 = cv2.resize(img, (100, 100))
        tensor = torch.from_numpy(img1.transpose(2, 0, 1))

        label = self.label[1]

        # print(img)
        # print(label)

        # digit, label = self.mnist[i]
        # digit = transforms.ToTensor()(digit)
        # bsds_image = self._random_bsds_image()
        # patch = self._random_patch(bsds_image)
        # patch = patch.float() / 255
        # blend = torch.abs(patch - digit)
        return tensor.float(), label

    def __len__(self):
        return len(self.img)


class PainDatasets(Dataset):

    def __init__(self, img_dir, label_path, transform=None, target_transform=None):
        super(PainDatasets, self).__init__()

        self.img_labels = pd.read_csv(label_path, sep=" ")
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.convtensor = transforms.ToTensor()

    def __getitem__(self, i):
        
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[i, 0])

        '''
        read_image is not working:
        UserWarning: Failed to load image Python extension: libtorch_cuda_cu.so: cannot open shared object file: No such file or directory 
        warn(f"Failed to load image Python extension: {e}")
        '''
        # image = read_image(img_path)
        image = self.convtensor(Image.open(img_path))
        # image = transforms.RandomResizedCrop(100)(image)

        label = self.img_labels.iloc[i, 1]

        # # display image
        # img = T.ToPILImage()(image)
        # img.show()

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

        # print(img)
        # print(label)

        # digit, label = self.mnist[i]
        # digit = transforms.ToTensor()(digit)
        # bsds_image = self._random_bsds_image()
        # patch = self._random_patch(bsds_image)
        # patch = patch.float() / 255
        # blend = torch.abs(patch - digit)


    def __len__(self):
        return len(self.img_labels)

class FerImageFolder(Dataset):

    def __init__(self, imgs, labels, transform = None):
        super(FerImageFolder, self).__init__()

        self.img = imgs
        self.label = labels
        self.transform = transform

    def __getitem__(self, index):
        # img = read_image(self.img[i])

        # img = transforms.Resize(100)(img)

        # image = read_image(self.img[index])
        # image = transforms.RandomResizedCrop(100)(image)

        # if self.transform:
        # image = self.transform(image)

        img = cv2.imread(self.img[index], cv2.IMREAD_COLOR)
        self.transform = self.transform
        img1 = cv2.resize(img, (100, 100))
        tensor = torch.from_numpy(img1.transpose(2, 0, 1))

        label = self.label[index]

        return tensor.float(), label, index

    def __len__(self):
        return len(self.img)


# Define a custom iterator that restarts from the beginning
class TragetRestartableIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)