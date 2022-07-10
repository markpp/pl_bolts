import argparse
import os, sys
import cv2
import numpy as np

import torch
import pytorch_lightning as pl

from config import hparams

def create_datamodule(config):
    sys.path.append('../../')

    if 'sewer' in hparams.data:
        from datamodules.sewer_datamodule import SewerDataModule
        dm = SewerDataModule(data_dir=hparams.data,
                             dataset=hparams.dataset,
                             batch_size=hparams.batch_size,
                             image_size=hparams.image_shape[1],
                             in_channels=hparams.image_shape[0],
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

    output_dir = 'trained_models/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    experiment_name = "{}_{}_{}_{}".format(hparams.name,
                                           hparams.dataset,
                                           hparams.image_shape[1],
                                           hparams.image_shape[0])

    dm = create_datamodule(hparams)
    if dm == None:
        print("failed to create datamodule")
        exit()

    from pytorch_lightning.loggers import TensorBoardLogger
    tb_logger = TensorBoardLogger(hparams.log_dir, name=experiment_name, default_hp_metric=False)

    from pytorch_lightning.callbacks import ModelCheckpoint
    mc = ModelCheckpoint(dirpath=os.path.join("trained_models/",experiment_name), monitor='val_loss')


    from autoencoder import AutoEncoder
    learner = AutoEncoder(hparams)

    from pytorch_lightning import Trainer
    trainer = Trainer(gpus=hparams.gpus,
                      max_epochs=hparams.max_epochs,
                      check_val_every_n_epoch=2,
                      limit_val_batches=0.1,
                      #sync_batchnorm = True, # only necessary for multi-gpu training
                      #precision=16,
                      #gradient_clip_val=0.0, #0 means don’t clip.
                      benchmark=True,
                      logger=tb_logger, # NB! no logger lists when adding images
                      callbacks=[mc])

    trainer.fit(learner, dm)

    # ´manually save trained model
    torch.save(learner.vqvae2, os.path.join(output_dir,"model.pt"))
    # save configuration
    pl.core.saving.save_hparams_to_yaml(config_yaml=os.path.join(output_dir,'hparams.yaml'), hparams=hparams)
