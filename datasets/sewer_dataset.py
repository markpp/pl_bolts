import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

import os
from glob import glob
import cv2

class SewerImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform):
        print()
        self.image_list = sorted([y for y in glob(os.path.join(root_dir, '*.jpg'))])
        if not len(self.image_list)>0:
            print("did not find any files here: {}".format(os.path.join(root_dir, '*.jpg')))
        self.transform = transform

    def load_sample(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        img = self.load_sample(self.image_list[idx])

        sample = {'image':img}
        if self.transform:
            sample = self.transform(**sample)
            img = sample["image"]
        img = img.transpose((2, 0, 1))
        return img

    def __len__(self):
        return len(self.image_list)


class SewerLabelDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, dataset, transform):
        import csv
        file = open(os.path.join(root_dir,'{}_labels.csv'.format(dataset)))
        csv_reader = csv.reader(file, delimiter=';')
        next(csv_reader)

        self.image_list, self.wls = [], []
        for i, row in enumerate(csv_reader):
            img_name = row[0]
            wl = row[1].split('_')[1]
            if 'Null' in wl:
                wl = -100
            else:
                wl = float(wl)

            self.image_list.append(os.path.join(root_dir,dataset,img_name+'.jpg'))
            self.wls.append(wl)

        if not len(self.image_list)>0:
            print("did not find any files")

        if not len(self.image_list)==len(self.wls):
            print("number of images does not match number of labels")

        self.transform = transform

    def load_sample(self, idx):
        img = cv2.imread(self.image_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, self.wls[idx]

    def __getitem__(self, idx):
        img, wl = self.load_sample(idx)

        sample = {'image':img}
        if self.transform:
            sample = self.transform(**sample)
            img = sample["image"]
        img = img.transpose((2, 0, 1))
        wl = torch.as_tensor(wl, dtype=torch.float32)
        wl = torch.unsqueeze(wl, 0)
        return img, wl

    def __len__(self):
        return len(self.image_list)
