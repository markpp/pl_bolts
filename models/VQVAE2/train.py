import argparse
import os, sys
import cv2
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers

from config import hparams

def create_datamodule(config):
    sys.path.append('../../../datasets')

    if 'sewer' in hparams.data:
        from sewer.datamodule import SewerDataModule
        dm = SewerDataModule(data_dir=hparams.data,
                             dataset=hparams.dataset,
                             batch_size=hparams.batch_size,
                             image_size=hparams.image_size,
                             in_channels=hparams.in_channels,
                             n_workers=hparams.n_workers)
        dm.setup()
        return dm
    elif 'themal' in hparams.data:
        from harbour_datamodule import HarbourDataModule
        print("toto")
    else:
        print("no such dataset: {}".format(hparams.data))
        return None

if __name__ == '__main__':
    """
    Trains an autoencoder from patches of thermal imaging.

    Command:
        python main.py
    """

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--checkpoint", type=str,
                    default='trained_models/model.pt', help="path to the checkpoint file")
    args = vars(ap.parse_args())

    dm = create_datamodule(hparams)
    if dm == None:
        print("failed to create datamodule")
        exit()

    experiment_name = "{}_{}_{}_{}".format(hparams.name,
                                           hparams.dataset,
                                           hparams.image_size,
                                           hparams.in_channels)

    logger = loggers.TensorBoardLogger(hparams.save_dir, name=experiment_name)

    from autoencoder import AutoEncoder
    learner = AutoEncoder(hparams)

    trainer = Trainer(gpus=hparams.gpus,
                      max_epochs=hparams.max_epochs,
                      check_val_every_n_epoch=2,
                      limit_val_batches=0.1,
                      sync_batchnorm = True, # only necessary for multi-gpu training
                      #precision=16,
                      gradient_clip_val=0.0, #0 means don’t clip.
                      benchmark=True,
                      logger=logger)

    trainer.fit(learner, dm)

    output_dir = 'trained_models/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    torch.save(learner.vqvae2, os.path.join(output_dir,"autoencoder.pt"))
