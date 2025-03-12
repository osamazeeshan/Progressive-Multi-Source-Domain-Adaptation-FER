from ast import arg
from ctypes import util
from operator import concat
import os
from pickle import NONE
import tarfile

import torch
import torchvision
import numpy as np

from PIL import Image
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST, USPS
from torchvision.transforms import Compose, ToTensor
from torchvision import transforms, datasets
from torchvision.io import read_image

from datasets.dataset import MNISTM, FerDatasets, PainDatasets
from utils.transforms import GrayscaleToRgb

import matplotlib.pyplot as plt

from torchvision.utils import make_grid
from utils.reproducibility import get_default_seed

import random

import config

SPLIT = 0.90

class BaseDataset():
    """
    Dataset base class: all the other dataloader classes will be inherit from base class 
    """

    def mnist_dataset(batch_size):
        try: 
            dataset = MNIST(config.CURRENT_DIR + config.DATASET_FOLDER + '/mnist', train=True, download=True,
                            transform=Compose([GrayscaleToRgb(), ToTensor()]))

            shuffled_indices = np.random.permutation(len(dataset))
            train_idx = shuffled_indices[:int(0.8*len(dataset))]
            val_idx = shuffled_indices[int(0.8*len(dataset)):]

            train_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True,
                                    sampler=SubsetRandomSampler(train_idx),
                                    num_workers=1, pin_memory=True)
            val_loader = DataLoader(dataset, batch_size=batch_size, drop_last=False,
                                    sampler=SubsetRandomSampler(val_idx),
                                    num_workers=1, pin_memory=True)

            return train_loader, val_loader

        except Exception as e:
            print("Error: ", e) 


    def load_mnist_dataset(batch_size):
        try:
            source_dataset = MNIST(config.CURRENT_DIR + config.DATASET_FOLDER + 'mnist', train=True, download=True,
                          transform=Compose([GrayscaleToRgb(), ToTensor()]))
            source_loader = DataLoader(source_dataset, batch_size=int(batch_size / 2),
                               shuffle=True, num_workers=1, pin_memory=True)
            
            return source_loader, source_dataset

        except Exception as e:
            print("Error: ", e)

    def load_mnistm_dataset(batch_size):
        try:
            target_dataset = MNISTM(train=False)
            target_loader = DataLoader(target_dataset, batch_size=int(batch_size / 2),
            shuffle=True, num_workers=1, pin_memory=True)

            return target_loader, target_dataset

        except Exception as e:
            print("Error: ", e)

    def usps_dataset(batch_size):
        try: 
            dataset = USPS(config.CURRENT_DIR + config.DATASET_FOLDER + '/usps', train=True, download=True,
                            transform=Compose([GrayscaleToRgb(), ToTensor()]))

            shuffled_indices = np.random.permutation(len(dataset))
            train_idx = shuffled_indices[:int(0.8*len(dataset))]
            val_idx = shuffled_indices[int(0.8*len(dataset)):]

            print(len(shuffled_indices))
            print(len(train_idx))
            print(len(val_idx))

            train_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True,
                                    sampler=SubsetRandomSampler(train_idx),
                                    num_workers=1, pin_memory=True)
            val_loader = DataLoader(dataset, batch_size=batch_size, drop_last=False,
                                    sampler=SubsetRandomSampler(val_idx),
                                    num_workers=1, pin_memory=True)

            print(type(train_loader))

            return train_loader, val_loader

        except Exception as e:
            print("Error: ", e) 

    def load_usps_dataset(batch_size):
        try:
            source_dataset = USPS(config.CURRENT_DIR + config.DATASET_FOLDER + '/usps', train=True, download=True,
                          transform=Compose([GrayscaleToRgb(), ToTensor()]))
            source_loader = DataLoader(source_dataset, batch_size=int(batch_size / 2),
                               shuffle=True, num_workers=1, pin_memory=True)
            
            return source_loader, source_dataset

        except Exception as e:
            print("Error: ", e)      


    def split_train_val(dataset_path, batch_size):

        dataset_train, dataset_val = BaseDataset.fer_dataloader_from_folders(dataset_path, split = True)

        train_set, val_set = torch.utils.data.random_split(dataset_train, [10771, 1500])
        print(len(train_set))
        print(len(val_set))

        data_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True, pin_memory=True)
        val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)
        # print(data_loader)

        return data_loader, val_loader

    def load_raf_dataset(dataset_path, batch_size):
        try:
            dataset_train, dataset_val  = BaseDataset.fer_dataloader_from_folders(dataset_path, split = True)

            dataset_loader = DataLoader(dataset_train, batch_size=int(batch_size / 2),
                               shuffle=True, num_workers=1, pin_memory=True)
            
            return dataset_loader, dataset_train

        except Exception as e:
            print("Error: ", e)

    def load_fer_dataset(dataset_path, batch_size):
        try:
            dataset_train, dataset_val  = BaseDataset.fer_dataloader_from_folders(dataset_path)

            dataset_loader = DataLoader(dataset_train, batch_size=int(batch_size / 2),
                               shuffle=True, num_workers=1, pin_memory=True)
            
            return dataset_loader, dataset_train

        except Exception as e:
            print("Error: ", e)        


    def fer_dataloader_from_folders(dataset_path, split = False):
        try:
            labels = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']
            label_mapping = {'surprise': 0, 'fear': 1, 'disgust': 2, 'happy': 3, 'sad': 4, 'angry': 5, 'neutral': 6}

            imgs_path_train, imgs_labels_train,imgs_path_val, imgs_labels_val = [], [], [], []

            for label in labels:
                label = str.lower(label)    # in case classes are not in lower caps; comment this line 
                class_file_pth_train = dataset_path + '/' + label
                for files in os.listdir(class_file_pth_train):
                    abs_pth = class_file_pth_train + '/' + files
                    imgs_path_train.append(abs_pth)
                    imgs_labels_train.append(label_mapping.get(label))

                if not split:
                    continue

                class_file_pth_val = dataset_path + '_test/' + label
                for files in os.listdir(class_file_pth_val):
                    abs_pth = class_file_pth_val + '/' + files
                    # print(abs_pth)
                    imgs_path_val.append(abs_pth)
                    imgs_labels_val.append(label_mapping.get(label))
            
            data_set_train = FerDatasets(imgs_path_train, imgs_labels_train)
            data_set_val = FerDatasets(imgs_path_val, imgs_labels_val)

            # print(len(imgs_path_val))
            # print(len(imgs_path_train))
            # print(type(imgs_path))
            # print(type(imgs_labels))

            return data_set_train, data_set_val

        except Exception as e:
            print("Error: ", e)

    def show_img(data_loader):

        images, foo = next(iter(data_loader))
        # npimg = make_grid(images, normalize=True, pad_value=.5).numpy()
        # fig, ax = plt.subplots(figsize=((13, 5)))
        # ax.imshow(np.transpose(npimg, (1, 2, 0)))
        # plt.setp(ax, xticks=[], yticks=[])

        # b, c, nx, ny = images.shape
        # if b == 3:
        im = images.squeeze()[0,0,:,:]
        # else:
        #     im = img.reshape((nx,ny))

        # convert image to PIL image and save it
        image_pil = torchvision.transforms.ToPILImage()(im)
        image_pil.save('01_image.png')

        plt.imshow(im)
        plt.show()

        return im

    def load_fer_data(root_path, batch_size, phase, split = False, root_path_2 = None):
        concatDatasets = []  

        ''' 
            1. transforms.Compose data augmentation DOES NOT INCREASE the size of the dataseta but rather apply augmentation on each image
            in a batch and which was then passed to the model.
            
            2. To increase the size of the datasets; need to apply augmentation(s) and then concatenate all the augmented and original 
            images together then passed to the data loader 
            
            ** Will apply first which randomly choose augmentation on each batch while training 
        '''         
         
        transform_dict = {
            'src': transforms.Compose(
            [
                # transforms.CenterCrop(100),
                transforms.Resize((100,100)),
                transforms.RandomHorizontalFlip(),
                # add augmentation for subject-vice adaptation 5 augmentation included RandomHorizontalFlip
                # transforms.RandomRotation(90),
                # transforms.RandomInvert(),
                # transforms.ColorJitter(brightness=.5, hue=.3),
                # transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.3,5)),

                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Grayscale(num_output_channels=3),
            ]),
            'tar': transforms.Compose(
            [
                # transforms.CenterCrop(100),
                transforms.Resize((100,100)),
                # transforms.RandomResizedCrop(100),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Grayscale(num_output_channels=3),
            ])}

        if(root_path_2 is None):
            data = datasets.ImageFolder(root=root_path, transform=transform_dict[phase])
        else:
            concatDatasets.append(datasets.ImageFolder(root=root_path, transform=transform_dict[phase]))
            concatDatasets.append(datasets.ImageFolder(root=root_path_2, transform=transform_dict[phase]))
            data = torch.utils.data.ConcatDataset(concatDatasets)

        # train_set, val_set = torch.utils.data.random_split(data, [26709, 2000]) if split else (data, None)
        train_set, val_set = torch.utils.data.random_split(data, [len(data) - int((1-SPLIT)*len(data)), int((1-SPLIT)*len(data))]) if split else (data, None)
        
        print(len(train_set))
        # print(len(val_set))

        data_loader_train = DataLoader(train_set, batch_size=batch_size, shuffle=phase=='src', drop_last=phase=='tar', num_workers=4)
        
        # data_loader_train = DataLoader(train_set, shuffle=False, drop_last=False, num_workers=1, pin_memory=True)
        data_loader_val = DataLoader(val_set, batch_size=1, shuffle=phase=='src', drop_last=phase=='tar', num_workers=4) if split else None

        # for i in range(1, 100):
        #     BaseDataset.show_img(data_loader_train)
        #     plt.show()
        # inp, tar = data.imgs[100]
        # img = read_image(inp)
        # f = plt.figure() 
        # print(img.shape)
        # b, nx, ny = img.shape
        # if b == 3:
        #     im = img.squeeze()[0,:,:]
        # else:
        #     im = img.reshape((nx,ny))
        # plt.imshow(im) 
            
        # # plt.imshow(img.squeeze()[0,:,:]) 
        # plt.show() 

        return data_loader_train, data_loader_val
    
    def load_multi_src_data(root_path, batch_size, phase, split = False):
        concatDatasets = []  

        ''' 
            1. transforms.Compose data augmentation DOES NOT INCREASE the size of the dataseta but rather apply augmentation on each image
            in a batch and which was then passed to the model.
            
            2. To increase the size of the datasets; need to apply augmentation(s) and then concatenate all the augmented and original 
            images together then passed to the data loader 
            
            ** Will apply first which randomly choose augmentation on each batch while training 
        '''         
         
        transform_dict = {
            'src': transforms.Compose(
            [
                # transforms.CenterCrop(100),
                transforms.Resize((100,100)),
                transforms.RandomHorizontalFlip(),
                # add augmentation for subject-vice adaptation 5 augmentation included RandomHorizontalFlip
                # transforms.RandomRotation(90),
                # transforms.RandomInvert(),
                # transforms.ColorJitter(brightness=.5, hue=.3),
                # transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.3,5)),

                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Grayscale(num_output_channels=3),
            ]),
            'tar': transforms.Compose(
            [
                # transforms.CenterCrop(100),
                transforms.Resize((100,100)),
                # transforms.RandomResizedCrop(100),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Grayscale(num_output_channels=3),
            ])}

        data_loader_train = []
        data_loader_val = []
        for path in root_path:
            data = datasets.ImageFolder(root=path, transform=transform_dict[phase])
            concatDatasets.append(datasets.ImageFolder(root=path, transform=transform_dict[phase]))

            # train_set, val_set = torch.utils.data.random_split(data, [26709, 2000]) if split else (data, None)
            train_set, val_set = torch.utils.data.random_split(data, [len(data) - int((1-SPLIT)*len(data)), int((1-SPLIT)*len(data))]) if split else (data, None)
            data_loader_train.append(DataLoader(train_set, batch_size=batch_size, shuffle=phase=='src', drop_last=phase=='tar', num_workers=4))
            data_loader_val.append(DataLoader(val_set, batch_size=1, shuffle=phase=='src', drop_last=phase=='tar', num_workers=4) if split else None)
        
        concat_data = torch.utils.data.ConcatDataset(concatDatasets)
        train_set, val_set = torch.utils.data.random_split(concat_data, [len(concat_data) - int((1-SPLIT)*len(concat_data)), int((1-SPLIT)*len(concat_data))]) if split else (concat_data, None)
        concat_data_loader_train = DataLoader(train_set, batch_size=batch_size, shuffle=phase=='src', drop_last=phase=='tar', num_workers=4)
        concat_data_loader_val = DataLoader(val_set, batch_size=1, shuffle=phase=='src', drop_last=phase=='tar', num_workers=4) if split else None
        
        return data_loader_train, data_loader_val, concat_data_loader_train, concat_data_loader_val


    def combine_datasets(data_loaders, batchSize, dataShuffle, tar_index):
        i = 0
        concat_datasets = []
        while i < tar_index:
            concat_datasets.append(data_loaders[i].dataset)
            i = i + 1
        
        multi_datasets = torch.utils.data.ConcatDataset(concat_datasets)    
        return DataLoader(multi_datasets, batch_size=batchSize, shuffle=dataShuffle, drop_last=False, num_workers=4)

    def show_im(data_loader):

        images, foo = next(iter(data_loader))

        from torchvision.utils import make_grid
        npimg = make_grid(images, normalize=True, pad_value=.5).numpy()
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=((13, 5)))
        import numpy as np
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.setp(ax, xticks=[], yticks=[])

        return fig, ax

    
    def load_target_data(target_dataset, batch_size, split = False):
        concatDatasets = []  

        transform_dict = {
            'src': transforms.Compose(
            [
                transforms.Resize((100,100)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Grayscale(num_output_channels=3),
            ]),
            'tar': transforms.Compose(
            [
                transforms.Resize((100,100)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Grayscale(num_output_channels=3),
            ])}
        
        # data = BaseDataset.convert_to_tensors(data)
        # concatDatasets.append(data)
        # concatDatasets.append(target_dataset)
        # data = torch.utils.data.ConcatDataset(concatDatasets)

        # train_set, val_set = torch.utils.data.random_split(data, [26709, 2000]) if split else (data, None)
        train_set, val_set = torch.utils.data.random_split(target_dataset, [len(target_dataset) - int((1-SPLIT)*len(target_dataset)), int((1-SPLIT)*len(target_dataset))]) if split else (target_dataset, None)
        
        data_loader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
        
        data_loader_val = DataLoader(val_set, batch_size=1, shuffle=True, drop_last=False) if split else None

        return data_loader_train, data_loader_val

    def convert_to_tensors(dataset):
        tensor_data = []
        for data in dataset:
            tensor_data = tensor_data + [(data[0], torch.tensor(data[1]))] if len(tensor_data) > 0 else [(data[0], torch.tensor(data[1]))]
        return tensor_data


    def expand_target_dataset(tar_data, tar_gt_label, src_data, batch_size, tar_pl=None, tar_prob_data=None):
        tar_size = len(tar_data)
        src_size = len(src_data) * batch_size
        multiplier = src_size // tar_size
        remainder = src_size % tar_size

        # Create indices for the expanded dataset
        indices = list(range(tar_size)) * multiplier + list(range(remainder))

        tar_exp_data = np.concatenate([tar_data for _ in range(multiplier)] + [tar_data[:remainder]])
        tar_exp_gt_labels = np.concatenate([tar_gt_label for _ in range(multiplier)] + [tar_gt_label[:remainder]])

        tar_exp_tar_pl = np.concatenate([tar_pl for _ in range(multiplier)] + [tar_pl[:remainder]]) if tar_pl is not None else None
        tar_exp_tar_prob = np.concatenate([tar_prob_data for _ in range(multiplier)] + [tar_prob_data[:remainder]]) if tar_prob_data is not None else None

        # tar_expanded_dataset = Subset(tar_data, indices)

        return {
            "_data_arr": tar_exp_data,
            "_gt_arr": tar_exp_gt_labels,
            "_label_arr": tar_exp_tar_pl,
            "_prob_arr": tar_exp_tar_prob
        }
    
        # return tar_exp_data, tar_exp_gt_labels, tar_exp_tar_pl, tar_exp_tar_prob

    def load_pain_dataset(dataset_path, label_path_train, label_path_val, batch_size, phase):
        try: 
            val_idx = []
            train_idx = []
            shuffle = False

            transform_dict = {
            'src': transforms.Compose(
            [
                # transforms.CenterCrop(100),
                transforms.Resize((96,96), antialias=True),   # Biovid=100,100 , UNBC= 96,96
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(90),
                # transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # transforms.Grayscale(num_output_channels=3),
            ]),
            'tar': transforms.Compose(
            [
                transforms.Resize((96,96), antialias=True),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(90),
                # transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # transforms.Grayscale(num_output_channels=3),
            ])}

            dataset_train = PainDatasets(dataset_path, label_path_train, transform=transform_dict[phase])
            dataset_size = len(dataset_train)
            
            """
                condition to split train and val if no validation is defined
            """
            if label_path_val is not None:
                dataset_val = PainDatasets(dataset_path, label_path_val)
                shuffle = True
            else: # for biovid
                np.random.seed(get_default_seed())
                shuffled_indices = np.random.permutation(len(dataset_train))
                train_idx = shuffled_indices[:int(0.70*len(dataset_train))]
                val_idx = shuffled_indices[int(0.70*len(dataset_train)):int(0.80*len(dataset_train))]
                test_idx = shuffled_indices[int(0.80*len(dataset_train)):int(1.0*len(dataset_train))]
                
                # num_train_samples = int(dataset_size * 0.70)
                # num_val_samples = int(dataset_size * 0.10)
                # num_test_samples = int(dataset_size * 0.20)

                # train_idx = list(range(num_train_samples))
                # val_idx = list(range(num_train_samples, num_train_samples + num_val_samples))
                # test_idx = list(range(num_train_samples + num_val_samples, dataset_size))

                # print(shuffled_indices)
                print("Train :", len(train_idx))
                print("Val: ", len(val_idx))
                print("Test: ", len(test_idx))
                # shuffle = True

                dataset_val = dataset_train

            train_loader = DataLoader(  dataset_train,
                                        batch_size=batch_size, 
                                        drop_last=True,
                                        num_workers=1, 
                                        pin_memory=True,
                                        shuffle=shuffle,
                                        sampler=SubsetRandomSampler(train_idx) if len(train_idx) != 0 else None
                                    )

            val_loader = DataLoader(    dataset_val, 
                                        batch_size=1, 
                                        drop_last=False, 
                                        num_workers=1, 
                                        pin_memory=True,
                                        sampler=SubsetRandomSampler(val_idx) if len(val_idx) != 0 else None
                                    )

            test_loader = DataLoader(   dataset_train, 
                                        batch_size=1, 
                                        drop_last=False, 
                                        num_workers=1, 
                                        pin_memory=True,
                                        sampler=SubsetRandomSampler(test_idx) if len(test_idx) != 0 else None
                                    )                        

            # test_loader = DataLoader(   dataset_val, 
            #                             batch_size=1, 
            #                             drop_last=False, 
            #                             num_workers=1, 
            #                             pin_memory=True,
            #                             sampler=SubsetRandomSampler(val_idx) if len(val_idx) != 0 else None
            #                         )

            # for i in range(1, 100):
            #     BaseDataset.show_img(train_loader)
            #     # plt.show()

            return train_loader, val_loader, test_loader

        except Exception as e:
            print("Error: ", e)        

    def generate_random_src_samples(src_dataloader, random_samples):
        total_samples = len(src_dataloader.dataset)
        
        # Ensure we're not asking for more samples than available
        random_samples = min(random_samples, total_samples)
        
        # Generate random indices
        random_indices = random.sample(range(total_samples), random_samples)

        dataset_subset = Subset(src_dataloader.dataset, random_indices)
        
        sampler = SubsetRandomSampler(random_indices)
        # Create a new dataloader with this sampler
        random_src_sample_loader = DataLoader(
            dataset_subset,
            batch_size=src_dataloader.batch_size,
            # sampler=sampler,
            num_workers=src_dataloader.num_workers,
            collate_fn=src_dataloader.collate_fn
        )
        
        return random_src_sample_loader
    
    def generate_random_src_samples_combine(relv_src_dataloader, src_dataloader, random_samples):
        prev_srcs_loader = BaseDataset.generate_random_src_samples(src_dataloader, random_samples)
        if relv_src_dataloader:
            relv_src_data = ConcatDataset([relv_src_dataloader.dataset, prev_srcs_loader.dataset])

            # Create a new dataloader with this sampler
            relv_src_dataloader = DataLoader(
                relv_src_data,
                batch_size=src_dataloader.batch_size,
                num_workers=src_dataloader.num_workers,
                collate_fn=src_dataloader.collate_fn
            )
        else:
            relv_src_dataloader = prev_srcs_loader
        
        return relv_src_dataloader