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

from transforms.normalization import normalize, unnormalize

output_dir = 'output/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

def predict(data_dir, dataset, model, crop_size=None, save=True, show=False):

    file = open(os.path.join(data_dir,'{}_labels.csv'.format(dataset)))
    csv_reader = csv.reader(file, delimiter=';')
    next(csv_reader)

    filenames, embs, wls = [], [], []
    for i, row in enumerate(csv_reader):
        img_name = row[0]
        wl = row[1].split('_')[1]
        if 'Null' in wl:
            wl = -100
        else:
            wl = float(wl)

        img_path = os.path.join(data_dir,dataset,img_name+'.jpg')
        #print(img_path)
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
        input = torch.as_tensor(input)/255.0
        input = normalize(input)

        rec, diffs, emb, code, recs = model(input.unsqueeze(0))

        #print(len(code))
        #print(code[0].numpy().shape)
        #print(emb[0].numpy().shape)
        #print(emb[1].numpy().shape)

        embs.append(emb[0].numpy().flatten())
        wls.append(wl)
        filenames.append(img_name)

        if i % 1000 == 0:
            print("row {}".format(i))
            break

        if show:
            rec = unnormalize(rec[0])
            rec = rec.mul(255).permute(1, 2, 0).byte().numpy()
            rec = cv2.cvtColor(rec, cv2.COLOR_RGB2BGR)
            cv2.imwrite("rec.png",rec)
            break

    if save:
        np.save("output/filenames.npy", filenames)
        np.save("output/embs.npy", embs)
        np.save("output/wl.npy", wls)


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
                    default='../../models/VQVAE2/trained_models/autoencoder_mlp.pth', help="path to the checkpoint file")
    args = vars(ap.parse_args())

    from models.VQVAE2.util import load_model

    model = load_model(args['checkpoint'])
    #predict(args['dir'], dataset='mlp', model=model)
