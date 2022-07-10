import argparse
import os, sys
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import yaml
import cv2
from glob import glob
import csv

output_dir = 'output/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

def predict(data_dir, dataset, model, crop_size=256, save=False):

    file = open(os.path.join(data_dir,'{}_labels.csv'.format(dataset)))
    csv_reader = csv.reader(file, delimiter=';')
    next(csv_reader)

    filenames, projs, embs, wls = [], [], [], []
    for i, row in enumerate(csv_reader):
        img_name = row[0]
        wl = row[1].split('_')[1]
        if 'Null' in wl:
            wl = 0
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

        proj, emb = model(input.unsqueeze(0), return_embedding=True)

        #print(proj[0].numpy().shape)
        #print(proj[0].numpy())
        #print(emb[0].shape)

        projs.append(proj[0].numpy())
        embs.append(emb[0].numpy())
        wls.append(wl)
        filenames.append(img_name)

        if i % 1000 == 0:
            print("row {}".format(i))

    np.save("output/filenames.npy", filenames)
    np.save("output/projs.npy", projs)
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
                    default='/home/aau/github/data/sewer/paper', help="path to data dir")
    ap.add_argument("-c", "--config", type=str,
                    default='config/byol.yaml', help="path to the config file")
    ap.add_argument("-p", "--checkpoint", type=str,
                    default='trained_models/model_mlp.pt', help="path to the checkpoint file")
    args = vars(ap.parse_args())

    with open(args['config'], 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    with torch.no_grad():
        from torchvision import models
        resnet = models.resnet50(pretrained=True)

        from byol_pytorch import BYOL
        model = BYOL(net=resnet,
                     image_size=config['exp_params']['image_size'],
                     hidden_layer=config['model_params']['hidden_layer'],
                     projection_size=config['model_params']['projection_size'],
                     projection_hidden_size=config['model_params']['projection_hidden_size'],
                     moving_average_decay=config['model_params']['moving_average_decay'])

        model = torch.load(args['checkpoint'])
        model.eval()


        predict(args['dir'], dataset='mlp', model=model)
