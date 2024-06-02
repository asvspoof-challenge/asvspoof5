import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import librosa
from random import randrange
import random


def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _,key,_,_,_,label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list
    elif is_eval:
        for line in l_meta:
            key= line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _,key,_,_,_,label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


def pad(x, max_len=64000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64000):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class Dataset_train(Dataset):
    def __init__(self,list_IDs, labels, base_dir):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64000 # take ~4 sec audio 
        self.fs  = 16000
     
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = librosa.load(self.base_dir+'flac_T/'+key+'.flac', sr=16000)  
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        target = self.labels[key] 
        return x_inp, target

class Dataset_dev(Dataset):
    def __init__(self,list_IDs, labels,base_dir):
        self.list_IDs = list_IDs
        
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64000 # take ~4 sec audio
        self.fs  = 16000

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = librosa.load(self.base_dir+'flac_D/'+key+'.flac', sr=16000) 
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        target = self.labels[key] 
        return x_inp, target


class Dataset_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64000

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = librosa.load(self.base_dir+'flac_E/'+key+'.flac', sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key
