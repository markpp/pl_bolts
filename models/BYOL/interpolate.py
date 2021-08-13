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

from datasets import get_dataset, get_dataloader
from dnn.models.ALAE import StyleALAE

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

def preprocess(img_path):
    if os.path.isfile(img_path):
        print(img_path)
    else:
        print("{} not found".format(print(img_path)))
    img = cv2.imread(img_path)

    x = largest_center_crop(img)
    x = cv2.resize(x, (resolution, resolution))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = x / 127.5 - 1
    x = torch.as_tensor(x, dtype=torch.float32)
    #x = normalize(x)
    #print(x.unsqueeze(0).shape)
    return x.unsqueeze(0)

if __name__ == '__main__':
    """
    Trains an autoencoder from patches of thermal imaging.

    Command:
        python main.py
    """

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str,
                    default='/home/aau/github/SimplePytorch-ALAE/results/sewer_results/StyleALAE-z-256_w-256_prog-(4,256)-(8,256)-(16,128)-(32,128)-(64,64)-(64,32)/checkpoints/ckpt_final.pt', help="path to the checkpoint file")
    ap.add_argument("-c", "--config", type=str,
                    default='/home/aau/github/SimplePytorch-ALAE/results/sewer_results/StyleALAE-z-256_w-256_prog-(4,256)-(8,256)-(16,128)-(32,128)-(64,64)-(64,32)/checkpoints/cfg.pt', help="path to the config file")
    ap.add_argument("-s", "--source", type=str,
                    default='/home/aau/github/data/sewer/all/00006735.png', help="path to the source file")
    ap.add_argument("-d", "--dest", type=str,
                    default='/home/aau/github/data/sewer/all/00024778.png', help="path to the dest file")
    args = vars(ap.parse_args())

    with torch.no_grad():

        if os.path.exists(args['config']):
            print("Overriding model config from file...")
            config = torch.load(args['config'])
        else:
            print("Config file not found")

        model = StyleALAE(model_config=config, device="cpu")
        model.load_train_state(args['model'])

        resolution = config['resolutions'][-1]

        A = preprocess(args['source'])
        a = model.encode(A,final_resolution_idx=5, alpha=1.0)[0]

        B = preprocess(args['dest'])
        b = model.encode(B,final_resolution_idx=5, alpha=1.0)[0]

        diff = b - a

        inter = np.linspace(0.0, 1.0, num=20)

        for idx,i in enumerate(inter):
            w = a + i * diff

            rec = model.decode(w.unsqueeze(0),final_resolution_idx=5, alpha=1.0)[0]

            rec = rec * 0.5 + 0.5
            rec = torch.clamp(rec, min=0.0, max=1.0)
            rec = rec.mul(255).permute(1, 2, 0).byte().numpy()
            rec = cv2.cvtColor(rec, cv2.COLOR_RGB2BGR)


            cv2.imwrite("interpolations/GAN_{}_A_{:.2f}-B_{:.2f}.jpg".format(idx,1-i,i),cv2.resize(rec, (256, 256)))
            #cv2.imshow("img",img)
            #key = cv2.waitKey()
            #if key == 27:
            #    break


            #print(row[4])
            #cv2.putText(img, "gt {:.0f}, pred {:.0f}".format(gt,pred), (x+10,y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            #cv2.imshow("test",img)
