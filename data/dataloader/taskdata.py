import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import csv
from scipy.io import loadmat

from torch.utils.data import Dataset, DataLoader, ConcatDataset

class AircraftData(Dataset):

    def __init__(self, path, anno, transform=None):
        self.path = path
        self.anno = anno
        with open(self.path + self.anno) as f:
            lines = [line.strip() for line in f ]
        
        self.classes = []
        self.files = []
        for line in lines:
            imgName = line[:7]
            label = line[8:]
            p = {}
            p['img_path'] = self.path + 'images/' + str(imgName) + '.jpg'
            p['label'] = label
            self.files.append(p)
            if label not in self.classes:
                self.classes.append(label)
        self.classes = sorted(self.classes)
        self.transform = transform
        self.max_classes = len(self.classes)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        label = self.classes.index(self.files[idx]['label'])
        
        img_path = self.files[idx]['img_path']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        image = image.numpy()
        image = image[:,:-20,:]
        
        return image.astype('float32'), label
    
class CarsData(Dataset):

    def __init__(self, path, anno, trainval, transform=None):
        self.path = path
        self.anno = anno
        items = loadmat(self.path + self.anno)
        
        self.files = []
        self.max_classes = 0
        for item in items['annotations'][0]:
            info = [row.flat[0] for row in item]
            if info[-1] == trainval:
                imgName = info[0]
                label = info[-2]
                p = {}
                p['img_path'] = self.path + str(imgName)
                p['label'] = label - 1
                self.files.append(p)
                if label > self.max_classes:
                    self.max_classes = label
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        label = self.files[idx]['label']
        img_path = self.files[idx]['img_path']
        image = Image.open(img_path).convert('RGB') 
        if self.transform:
            image = self.transform(image)
        image = image.numpy()
        
        return image.astype('float32'), np.int64(label)
    
class DogsData(Dataset):

    def __init__(self, path, anno, transform=None):
        self.path = path
        self.anno = anno
        items = loadmat(self.path + self.anno)
        
        self.files = []
        self.max_classes = 0
        for index in range(len(items['file_list'])):
            imgName = items['file_list'][index][0][0]
            label = items['labels'][index][0]
            p = {}
            p['img_path'] = self.path + 'Images/' + str(imgName)
            p['label'] = label - 1
            self.files.append(p)
            if label > self.max_classes:
                self.max_classes = label
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        label = self.files[idx]['label']
        img_path = self.files[idx]['img_path']
        image = Image.open(img_path).convert('RGB') 
        if self.transform:
            image = self.transform(image)
        image = image.numpy()
        
        return image.astype('float32'), np.int64(label)
    
class BirdsData(Dataset):

    def __init__(self, path, anno, trainval, transform=None):
        self.path = path
        self.anno = anno
        
        with open(self.path + self.anno) as f:
            lines = [line.strip() for line in f ]
        
        self.max_classes = 0
        self.files = []
        for line in lines:
            split = line.split(' ')
            if int(split[2]) == trainval:
                imgName = split[0]
                label = int(split[1])
                p = {}
                p['img_path'] = self.path + 'images/' + imgName
                p['label'] = label - 1
                self.files.append(p)
                if label > self.max_classes:
                    self.max_classes = label
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        label = self.files[idx]['label']
        img_path = self.files[idx]['img_path']
        image = Image.open(img_path).convert('RGB') 
        if self.transform:
            image = self.transform(image)
        image = image.numpy()
        
        return image.astype('float32'), np.int64(label)

class IndoorData(Dataset):

    def __init__(self, path, anno, transform=None):
        self.path = path
        self.anno = anno
        with open(self.path + self.anno) as f:
            lines = [line.strip() for line in f ]
        
        self.classes = []
        self.files = []
        for line in lines:
            label = line.split('/')[0]
            p = {}
            p['img_path'] = self.path + 'Images/' + line
            p['label'] = label
            self.files.append(p)
            if label not in self.classes:
                self.classes.append(label)
        self.classes = sorted(self.classes)
        self.transform = transform
        self.max_classes = len(self.classes)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        label = self.classes.index(self.files[idx]['label'])
        
        img_path = self.files[idx]['img_path']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        image = image.numpy()
        
        return image.astype('float32'), label