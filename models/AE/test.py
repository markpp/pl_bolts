import argparse
import os, sys
import cv2
import numpy as np
import csv

import torch
import pytorch_lightning as pl
#from torchsummary import summary
import torch.nn.functional as F
from torchvision import transforms

import yaml

#device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == "cuda:0" else "cpu")

def largest_center_crop(img):
    h, w, c = img.shape
    if h > w:
        top_h = int((h - w) / 2)
        img = img[top_h:top_h + w]
    else:
        left_w = int((w - h) / 2)
        img = img[:, left_w:left_w + h]
    return img


if __name__ == '__main__':
    """
    Trains an autoencoder from patches of thermal imaging.

    Command:
        python main.py
    """

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--encoder", type=str,
                    default='trained_models/encoder.pt', help="path to the encoder model")
    ap.add_argument("-d", "--decoder", type=str,
                    default='trained_models/decoder.pt', help="path to the decoder model")
    ap.add_argument("-c", "--config", type=str,
                    default='config/ae.yaml', help="path to the config file")
    args = vars(ap.parse_args())

    with open(args['config'], 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    data_dir = '/home/aau/github/data/sewer/paper/'
    output_dir = '/home/aau/github/waterlevel/ae/paper_mlp'

    file = open(os.path.join(data_dir,'mlp_labels.csv'))
    csv_reader = csv.reader(file,delimiter=';')
    next(csv_reader)

    with torch.no_grad():
        encoder = torch.load(args['encoder'])
        decoder = torch.load(args['decoder'])
        encoder.eval()
        decoder.eval()

        #from model import create_encoder, create_decoder
        #encoder = create_encoder(config)
        #decoder = create_decoder(config)

        filenames, zs, recs, wls = [], [], [], []
        #for img in test_dataset:
        for row in csv_reader:
            img_path = os.path.join(data_dir,'mlp',row[0]+'.jpg')

            img_name = row[0]
            wl = row[1].split('_')[1]
            if 'Null' in wl:
                wl = 0
            else:
                wl = int(wl)

            if os.path.isfile(img_path):
                img = cv2.imread(img_path)

                x = largest_center_crop(img)
                x = cv2.resize(x, (config['exp_params']['image_size'], config['exp_params']['image_size']))
                x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
                x = x.transpose((2, 0, 1))
                x = x / 127.5 - 1
                x = torch.as_tensor(x, dtype=torch.float32)
                #x = normalize(x)
                #print(x.unsqueeze(0).shape)
                x = x.unsqueeze(0)
                z = encoder(x)[0]
                rec = decoder(z.unsqueeze(0))[0]
                rec = rec * 0.5 + 0.5
                #rec = torch.clamp(rec, min=0.0, max=1.0)
                rec = rec.mul(255).permute(1, 2, 0).byte().numpy()
                rec = cv2.cvtColor(rec, cv2.COLOR_RGB2BGR)

                filenames.append(img_name)
                zs.append(z.flatten().tolist())
                #recs.append(rec)
                wls.append(wl)

                '''
                cv2.imshow("rec",rec)
                cv2.imshow("img",img)
                key = cv2.waitKey()
                if key == 27:
                    break
                '''

        #print(filenames)
        #print(zs)
        #print(recs)
        #print(wls)
        #'''

        np.save(os.path.join(output_dir,"filenames.npy"), filenames)
        np.save(os.path.join(output_dir,"zs.npy"), zs)
        #np.save(os.path.join(output_dir,"recs.npy"), recs)
        np.save(os.path.join(output_dir,"wl.npy"), wls)
        #'''
