import torch
from torch.utils.data import DataLoader
import torchvision
import pytorch_lightning as pl

import os, sys

sys.path.append('../')

class SewerDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, dataset, batch_size, image_size, in_channels, n_workers):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.batch_size = batch_size
        self.image_size = image_size
        self.in_channels = in_channels
        self.n_workers = n_workers

    def setup(self, stage=None):
        from transforms.augmentation import basic_augmentation

        if stage == 'fit' or stage is None:
            '''
            #from datasets.sewer_dataset import SewerImageDataset
            self.data_train = SewerImageDataset(root_dir=os.path.join(self.data_dir,self.dataset),
                                                transform=basic_augmentation(img_height=self.image_size,
                                                                             img_width=self.image_size,
                                                                             img_pad=30,
                                                                             in_channels=self.in_channels))
            self.data_val = SewerImageDataset(root_dir=os.path.join(self.data_dir,'validation'),
                                              transform=basic_augmentation(img_height=self.image_size,
                                                                           img_width=self.image_size,
                                                                           in_channels=self.in_channels))
            '''
            from datasets.sewer_dataset import SewerLabelDataset
            self.data_train = SewerLabelDataset(root_dir=self.data_dir,
                                                dataset=self.dataset,
                                                transform=basic_augmentation(img_height=self.image_size,
                                                                             img_width=self.image_size,
                                                                             img_pad=30,
                                                                             in_channels=self.in_channels))
            self.data_val = SewerLabelDataset(root_dir=self.data_dir,
                                              dataset='validation',
                                              transform=basic_augmentation(img_height=self.image_size,
                                                                           img_width=self.image_size,
                                                                           in_channels=self.in_channels))

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=self.n_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.n_workers, pin_memory=True, shuffle=False)


if __name__ == '__main__':

    import cv2
    from transforms.normalization import denormalize


    dm = SewerDataModule(data_dir='/home/aau/github/data/sewer/paper/',
                         dataset='mlp',
                         batch_size=16,
                         image_size=144,
                         in_channels=3,
                         n_workers=1)

    dm.setup()

    # cleanup output dir
    import os, shutil
    output_root = "output/"
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root)

    sample_idx = 0
    for batch_id, batch in enumerate(dm.train_dataloader()):
        if len(batch) == 1: # images
            imgs = batch
            for img in imgs:
                img = denormalize(img).mul(255).permute(1, 2, 0).byte().numpy()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                output_dir = os.path.join(output_root,str(batch_id).zfill(6))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                filename = "id-{}.png".format(str(sample_idx).zfill(6))
                cv2.imwrite(os.path.join(output_dir,filename),img)
                sample_idx = sample_idx + 1
        else:
            imgs, labels = batch
            for img, label in zip(imgs,labels):
                img = denormalize(img).mul(255).permute(1, 2, 0).byte().numpy()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                label = label.numpy()
                cv2.putText(img,"label: {:.2f}".format(label), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                output_dir = os.path.join(output_root,str(batch_id).zfill(6))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                filename = "id-{}.png".format(str(sample_idx).zfill(6))
                cv2.imwrite(os.path.join(output_dir,filename),img)
                sample_idx = sample_idx + 1
        if batch_id > 1:
            break
        '''
        else: # image pairs
            imgs, imgs_ = batch
            for img, img_ in zip(imgs,imgs_):
                img = img.mul(255).permute(1, 2, 0).byte().numpy()
                img_ = img_.mul(255).permute(1, 2, 0).byte().numpy()
                output_dir = os.path.join(output_root,str(batch_id).zfill(6))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                filename = "id-{}.png".format(str(sample_idx).zfill(6))
                cv2.imwrite(os.path.join(output_dir,filename),img[:,:,0])
                filename = "id-{}_.png".format(str(sample_idx).zfill(6))
                cv2.imwrite(os.path.join(output_dir,filename),img_[:,:,0])
                sample_idx = sample_idx + 1
        '''
