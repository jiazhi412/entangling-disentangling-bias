import torch
from torch.utils.data import Dataset
import numpy as np
import dataloader.Diabetes_data_utils as utils
import random

class DiabetesDataset_II(Dataset):
    def __init__(self, path, quick_load, bias_attr, middle_age, mode, idx=None):
        self.bias_attr = bias_attr
        self.features, self.labels_onehotv, self.bias_onehotv = self._load_data(path, quick_load)
        self.labels = self._onehotvector_to_label(self.labels_onehotv)
        self.bias = self.bias_onehotv
        self.idx = None

        # print('Note', 'YoungP', 'YoungN', 'OldP', 'OldN')
        np, nn, pp, pn = utils.split_by_age(self.features, self.labels, self.bias, middle_age)

        moderate_size = 20
        if "moderate" in mode and idx != None:
            if mode == 'eb1_moderate':
                np = list(set(np) - set(idx[0]))
                pn = list(set(pn) - set(idx[3]))
            if mode == 'eb2_moderate':
                nn = list(set(nn) - set(idx[1]))
                pp = list(set(pp) - set(idx[2]))
        m = min(len(pp), len(pn), len(np), len(nn))
        print('Inpu', len(np), len(nn), len(pp), len(pn))
        if mode == 'eb1':
            self._sift(pp+nn)
        elif mode == 'eb2':
            self._sift(np+pn)
        elif mode == 'eb1_balanced':
            self._sift(random.sample(pp, m) + random.sample(nn, m))
        elif mode == 'eb2_balanced':
            self._sift(random.sample(np, m) + random.sample(pn, m))
        elif mode == 'balanced':
            self._sift(random.sample(np, m) + random.sample(pn, m) + random.sample(pp, m) + random.sample(nn, m))
        elif mode == 'eb1_moderate':
            nn = random.sample(nn, len(nn)-moderate_size)
            pp = random.sample(pp, len(pp)-moderate_size)
            np = random.sample(np, moderate_size)
            pn = random.sample(pn, moderate_size)
            self._sift(pp+nn+np+pn)
            self.idx = [np, nn, pp, pn]
        elif mode == 'eb2_moderate':
            np = random.sample(np, len(np)-moderate_size)
            pn = random.sample(pn, len(pn)-moderate_size)
            nn = random.sample(nn, moderate_size)
            pp = random.sample(pp, moderate_size)
            self._sift(pp+nn+np+pn)
            self.idx = [np, nn, pp, pn]
        np, nn, pp, pn = utils.split_by_age(self.features, self.labels, self.bias, middle_age)
        print('Outp', len(np), len(nn), len(pp), len(pn))

    def __getitem__(self, i):
        feature = self.features[i]
        label = np.array(self.labels[i])
        bias = self.bias[i]
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