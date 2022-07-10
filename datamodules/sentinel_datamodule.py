import torch
from torch.utils.data import DataLoader
import torchvision
import pytorch_lightning as pl

import os, sys
import numpy as np

sys.path.append('../')

class SentinelDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, image_size, n_workers=1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        #self.mean, self.std = [0.3362, 0.3214, 0.3174], [0.2276, 0.2299, 0.2779]
        self.n_workers = n_workers

    def setup(self, stage=None):
        from transforms.augmentation import sample_augmentation

        if stage == 'fit' or stage is None:
            from datasets.sentinel_dataset import SentinelDataset

            self.data_train = SentinelDataset(os.path.join(self.data_dir,'sentinel-2'),
                                              transform=sample_augmentation(img_height=self.image_size,
                                                                            img_width=self.image_size))

            self.data_val = SentinelDataset(os.path.join(self.data_dir,'sentinel-2_test'),
                                            transform=sample_augmentation(img_height=self.image_size,
                                                                          img_width=self.image_size))

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=self.n_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.n_workers, pin_memory=True, shuffle=False)



if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import cv2
    from transforms.normalization import denormalize

    dm = SentinelDataModule(data_dir='/home/aau/github/data/redoco2/dynearthnet-challenge',
                            batch_size=16,
                            image_size=256,
                            n_workers=1)
    dm.setup()


    '''
    from utils.data import get_mean_std

    mean, std = get_mean_std(dm.train_dataloader())
    print("MEAN {}".format(mean.numpy()))
    print("STD {}".format(std.numpy()))

    '''

    # cleanup output dir
    import os, shutil
    output_root = "output/"
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root)

    sample_idx = 0

    for batch_id, batch in enumerate(dm.train_dataloader()):
        if len(batch) == 2:
            imgs, imgs_ = batch
            break
        else:
            imgs = denormalize(batch)
            for i, img in enumerate(imgs):
                for c, channel in enumerate(img):
                    #b, g, r = imgs[0,0], imgs[0,1], imgs[0,2]
                    #bgr = np.dstack((b, g, r))
                    plt.imshow(channel)
                    plt.savefig(os.path.join(output_root,'batch_{}_sample_{}_channel_{}.png'.format(batch_id,i,c)))
                    #plt.show()
            break
