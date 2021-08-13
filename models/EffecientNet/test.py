import argparse
import os

import torch
import pytorch_lightning as pl

from config import hparams
import numpy as np
import cv2
import json
import random

sys.path.append('../datamodule/')
from transforms import normalize

def test(model_path, json_list_path, show=False, save=False):
    with torch.no_grad():

        #from lightning_model import LightningClassifier
        #model = LightningClassifier(hparams)

        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_name(hparams.net)
        from torch import nn
        model._fc = nn.Linear(model._fc.in_features, hparams.output_size)

        model.load_state_dict(torch.load(model_path))
        model.eval()

        with open(json_list_path) as f:
            img_files = f.read().splitlines()

        mean_diff = []
        for idx, img_file in enumerate(img_files):
            print(img_file)
            img_file = img_file.replace('.json','.jpg')
            img = cv2.imread(img_file)
            img_h, img_w = img.shape[:2]

            with open(img_file.replace('.jpg','.json'), 'r') as f:
                json_labels = json.loads(f.read())

            json_label = json_labels[random.randint(0,len(json_labels))-1]
            top_point = json_label['keypoints']['top']
            x = int(img_w*top_point[0])-hparams.input_size//2
            y = int(img_h*top_point[1])-hparams.input_size//4

            crop = img[y:y+hparams.input_size,x:x+hparams.input_size]
            #img = img.copy()
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = crop.transpose((2, 0, 1))
            crop = crop.astype(np.float32) / 255.0
            crop = normalize(crop)
            crop = torch.from_numpy(crop)

            pred = model(crop.unsqueeze(0))[0].detach()
            pred = pred.numpy()[0]*100.0
            gt = json_label['flow']
            diff = abs(gt-pred)

            mean_diff.append(diff)

            if show or save:
                if diff > 5.0:
                    cv2.putText(img, "gt {:.0f}, pred {:.0f}".format(gt,pred), (x+10,y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    cv2.rectangle(img, (x,y), (x+hparams.input_size,y+hparams.input_size), (0,0,255), 2)
                else:
                    cv2.putText(img, "gt {:.0f}, pred {:.0f}".format(gt,pred), (x+10,y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv2.rectangle(img, (x,y), (x+hparams.input_size,y+hparams.input_size), (0,255,0), 2)

                if save:
                    cv2.imwrite("output/frame_{}.jpg".format(idx),img)
                if show:
                    cv2.imshow("output",img)
                    key = cv2.waitKey()
                    if key == 27:
                        break
        print(np.mean(mean_diff))

if __name__ == '__main__':
    """
    Trains an autoencoder from patches of thermal imaging.

    Command:
        python main.py
    """

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str,
                    default='trained_models/model_dict.pth', help="Path to trained model")
    ap.add_argument("-l", "--list", type=str,
                    default=None, help="Path to test image list")
    ap.add_argument("-sh", "--show", type=int,
                    default=0, help="show predictions?")
    ap.add_argument("-sa", "--save", type=int,
                    default=1, help="save predictions?")
    args = vars(ap.parse_args())

    test(args['model'], args['list'], show=args['show'], save=args['save'])
