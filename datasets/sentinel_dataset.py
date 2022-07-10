import torch
import torchvision
import pytorch_lightning as pl

import os
from glob import glob
import cv2
#import georasters as gr
#import rasterio
import tifffile as tf # documentation: https://github.com/cgohlke/tifffile
import numpy as np
import random

class SentinelDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.tif_list = sorted([y for y in glob(root_dir+'/*/*.tif', recursive=True)])
        self.tif_list = [x for x in self.tif_list if not 'visualization' in x]
        if not len(self.tif_list)>0:
            print("did not find any files")

        self.factor = 8192 #4096
        self.transform = transform

    def load_sample(self, path):
        #raster = rasterio.open(path)
        raster = tf.imread(path)[:,:,:3]/self.factor
        #print(path)
        #print(raster.shape)
        #print("min {}, max {}".format(raster.min(),raster.max()))

        #gdf = raster[0].to_geopandas()
        #gdf = gdf.to_crs(epsg=25832)
        #if len(raster.indexes) < 9:
        #    print(path)
        #    print(raster.indexes)

        return raster

    def __getitem__(self, idx):
        sample = self.load_sample(self.tif_list[idx])

        #b = sample.read(1)/self.factor
        #g = sample.read(2)/self.factor
        #r = sample.read(3)/self.factor

        #re1 = sample.read(4)/4096
        #re2 = sample.read(5)/4096
        #re3 = sample.read(6)/4096

        #img = np.dstack((b, g, r))#, re1, re2, re3))

        img = sample

        if self.transform:
            sample = self.transform(**{'image':img})
            img = sample["image"]

        img = img.transpose((2, 0, 1))
        return torch.as_tensor(img, dtype=torch.float32)

    def __len__(self):
        return len(self.tif_list)


if __name__ == '__main__':

    import matplotlib.pyplot as plt


    data_dir = '/home/aau/github/data/redoco2/sentinel/dynearthnet'

    dataset = SentinelDataset(os.path.join(data_dir,'sentinel-2'))

    idx = random.randint(0,len(dataset))

    sample = dataset[idx]

    print(sample.shape)
    print(sample.bounds)
    print(sample.crs)
    print(sample.indexes)

    #10 bands are: ['Blue', 'Green', 'Red', 'Red Edge 1', 'Red Edge 2', 'Red Edge 3', 'NIR', 'Red Edge 4', 'SWIR1', 'SWIR2']
    print(np.max(sample.read(1)))
    print(np.min(sample.read(1)))
    b = sample.read(1)/4096
    g = sample.read(2)/4096
    r = sample.read(3)/4096
    band4 = sample.read(4)/4096

    nrg = np.dstack((b, g, r))

    plt.imshow(nrg)

    #plt.imshow(band4, cmap='pink')

    plt.show()
