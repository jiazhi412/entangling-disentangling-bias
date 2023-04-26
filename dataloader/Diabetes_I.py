import torch
from torch.utils.data import Dataset
import numpy as np
import dataloader.Diabetes_data_utils as utils
import random

class DiabetesDataset_I(Dataset):
    def __init__(self, path, quick_load, bias_attr, middle_age, mode='train', train_ratio=0.8, balance=False, minority='young', minority_size=100, idx=None):
        self.bias_attr = bias_attr
        self.features, self.labels_onehotv, self.bias_onehotv = self._load_data(path, quick_load)
        self.labels = self._onehotvector_to_label(self.labels_onehotv)
        self.bias = self.bias_onehotv

        np, nn, pp, pn = utils.split_by_age(self.features, self.labels, self.bias, middle_age)
        if mode == 'train':
            np, nn, pp, pn = random.sample(np, int(len(np)*train_ratio)), random.sample(nn, int(len(nn)*train_ratio)), random.sample(pp, int(len(pp)*train_ratio)), random.sample(pn, int(len(pn)*train_ratio))
        elif mode == 'test':
            np = list(set(np) - set(idx[0]))
            nn = list(set(nn) - set(idx[1]))
            pp = list(set(pp) - set(idx[2]))
            pn = list(set(pn) - set(idx[3]))
        self.idx = [np, nn, pp, pn]
        
        # print('Note', 'YoungP', 'YoungN', 'OldP', 'OldN')
        # YoungP -> NP
        m = min(len(pp), len(pn), len(np), len(nn))
        print('Inpu', len(np), len(nn), len(pp), len(pn))
        if balance:
            np, nn, pp, pn = random.sample(np, m), random.sample(nn, m), random.sample(pp, m), random.sample(pn, m)
        if minority == "young":
            self._sift(pp + pn + random.sample(np, minority_size//2) + random.sample(nn, minority_size//2))
        elif minority == "old":
            self._sift(np + nn + random.sample(pp, minority_size//2) + random.sample(pn, minority_size//2))
        elif minority == None:
            self._sift(np + nn + pp + pn)
        np, nn, pp, pn = utils.split_by_age(self.features, self.labels, self.bias, middle_age)
        print('Outp', len(np), len(nn), len(pp), len(pn))

    def __getitem__(self, i):
        feature = self.features[i]
        label = np.array(self.labels[i])
        bias = self.bias[i]
        # return feature, label.float(), bias.float()
        return feature, label, bias

    def __len__(self):
        data_len = self.features.shape[0]
        return data_len
    
    def get_idx(self):
        return self.idx
    
    def _sift(self, idx):
        self.features = self.features[idx]
        self.labels = self.labels[idx]
        self.bias = self.bias[idx]

    def _load_data(self, path, quick_load):
        if quick_load:
            data = utils.quick_load(path)
        else:
            data = utils.data_processing(path)
        data = torch.tensor(data,dtype=torch.float)
        features = data[:,:data.shape[1]-2]
        labels = data[:,data.shape[1]-2:]
        bias = utils.get_bias(data, self.bias_attr)
        return features, labels, bias

    # [1,0] -> 1; [0,1] -> 0
    def _onehotvector_to_label(self, onehotvector):
        labels = []
        for v in onehotvector:
            for i in range(v.shape[0]):
                if v[i] == 0:
                    label = i
                    labels.append(label)
                    break
        labels = torch.tensor(labels).unsqueeze_(1)
        return labels