import argparse
import os, sys
import cv2
import numpy as np
import csv

import torch
import pytorch_lightning as pl
#from torchsummary import summary
import torch.nn.functional as F

sys.path.append(os.path.dirname(__file__))

from config import hparams
from VQVAE2 import VQVAE2

def load_model(path):

    if '.pth' in path:
        model = VQVAE2(in_channels=hparams.in_channels,
                       hidden_channels=hparams.hidden_channels,
                       embed_dim=hparams.embed_dim,
                       nb_entries=hparams.nb_entries,
                       nb_levels=hparams.nb_levels,
                       scaling_rates=hparams.scaling_rates)

        model.load_state_dict(torch.load(path))
    else:
        model = torch.load(path)

    model.eval()

    return model



if __name__ == '__main__':
    """
    Trains an autoencoder from patches of thermal imaging.

    Command:
        python main.py
    """

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--checkpoint", type=str,
                    default='trained_models/autoencoder.pt', help="path to the checkpoint file")
    args = vars(ap.parse_args())

    model = load_model(args['checkpoint'])


    #torch.save(model.state_dict(), args['checkpoint'].replace('.pt','.pth'))
