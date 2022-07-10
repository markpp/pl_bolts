import argparse
import os, sys
import cv2
import numpy as np
import csv

import torch
import pytorch_lightning as pl

import torch.nn.functional as F
from torchvision import transforms


sys.path.append('../../')

output_dir = '/home/aau/github/waterlevel/FastGAN/paper_mlp'


def predict(data_dir, dataset, model, crop_size=None, save=True, show=True):

    file = open(os.path.join(data_dir,'{}_labels.csv'.format(dataset)))
    csv_reader = csv.reader(file, delimiter=';')
    next(csv_reader)

    filenames, zs, wls = [], [], []
    for i, row in enumerate(csv_reader):
        img_name = row[0]
        wl = row[1].split('_')[1]
        if 'Null' in wl:
            wl = 0
        else:
            wl = int(wl)

        img_path = os.path.join(data_dir,dataset,img_name+'.jpg')
        img = cv2.imread(img_path)

        # center crop
        if crop_size:
            img_h, img_w = img.shape[:2]
            x = img_w//2-crop_size//2
            y = img_h//2-crop_size//2
            crop = img[y:y+crop_size,x:x+crop_size]
            input = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        else:
            input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input = input.transpose((2, 0, 1))
        input = input / 127.5 - 1
        input = torch.as_tensor(input, dtype=torch.float32)


        #rec, diffs, emb, code, recs = model(input.unsqueeze(0))
        real_imgs = torch.unsqueeze(input, 0).to(device) # (HxW -> CxHxW)
        rf, recs, part = net_D(real_imgs,label='real')
        rec_big, rec_small, rec_part = recs

        #print(len(code))
        print(rf.numpy().shape)


        zs.append(rf[0].cpu().numpy().flatten())
        wls.append(wl)
        filenames.append(img_name)

        if i % 1000 == 0:
            print("row {}".format(i))
            break

        if show:
            rec = rec_big[0].cpu() * 0.5 + 0.5
            rec = rec.mul(255).permute(1, 2, 0).byte().numpy()
            rec = cv2.cvtColor(rec, cv2.COLOR_RGB2BGR)
            cv2.imwrite("rec.png",rec)
            break

    if save:

        np.save(os.path.join(output_dir,"filenames.npy"), filenames)
        np.save(os.path.join(output_dir,"zs.npy"), zs)
        #np.save(os.path.join(output_dir,"recs.npy"), recs)
        np.save(os.path.join(output_dir,"wl.npy"), wls)

if __name__ == '__main__':
    """
    Test autoencoder using specific patches from raster image.

    Command:
        python main.py
    """

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", type=str,
                    default='../../../data/sewer/paper', help="path to data dir")
    ap.add_argument("-p", "--checkpoint", type=str,
                    default='../../models/FastGAN/train_results/sewer_test2/models/1000.pth', help="path to the checkpoint file")
    args = vars(ap.parse_args())

    device = torch.device('cuda:0')

    from models.FastGAN.models import Discriminator
    net_D = Discriminator(ndf=64, nc=3, im_size=256)
    checkpoint = torch.load(args['checkpoint'])
    net_D.load_state_dict(checkpoint['d'])
    net_D.to(device)

    with torch.no_grad():
        predict(args['dir'], dataset='mlp', model=net_D)
