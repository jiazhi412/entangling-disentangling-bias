# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os


import torch.utils.data as data
import torchvision.datasets as datasets


class WholeDataLoader(Dataset):
    def __init__(self,option, istrain=True):
        # self.data_split = istrain
        self.is_train = istrain
        data_dic = np.load(os.path.join(option.data_dir,f'mnist_10color_jitter_var_{option.color_var}.npy'),encoding='latin1', allow_pickle=True).item()
        if self.is_train == 1:
            self.image = data_dic['train_image'].astype(np.uint8)
            self.label = data_dic['train_label'].astype(np.uint8)
        else:
            self.image = data_dic['test_image'].astype(np.uint8)
            self.label = data_dic['test_label'].astype(np.uint8)

        color_var = option.color_var
        self.color_std = color_var**0.5
        self.toTensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5,0.5,0.5),
                                             (0.5,0.5,0.5))
        self.ToPIL = transforms.Compose([
                              transforms.ToPILImage(),
                              ])

    def __getitem__(self,index):
        label = self.label[index]
        image = self.image[index]

        image = self.toTensor(image)
        image = self.ToPIL(image)
        label_image = image.resize((14,14), Image.NEAREST) 
        label_image = torch.from_numpy(np.transpose(label_image,(2,0,1)).copy())
        # mask_image = torch.lt(label_image.float()-0.00001, 0.) * 255
        label_image = torch.div(label_image,32)
        label_image = label_image.long()
        colorlabel = label_image.view(3,-1).max(1)[0].long()
        
        bins = np.array([0, 4, 8])
        c1 = np.digitize(colorlabel[0], bins) - 1
        c2 = np.digitize(colorlabel[1], bins) - 1
        c3 = np.digitize(colorlabel[2], bins) - 1
        colorlabel = c1 * 2**2 + c2 * 2**1 + c3 * 2**0

        # COLOUR_MAP = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [225, 225, 0], [225, 0, 225],
        #             [0, 255, 255], [255, 128, 0], [255, 0, 128], [128, 0, 255], [128, 128, 128]]
        
        image = self.toTensor(image)
        image = self.normalize(image)
        return image, label.astype(np.long), colorlabel

    def __len__(self):
        return self.image.shape[0]