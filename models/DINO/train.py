import argparse
import os, sys
import cv2
import numpy as np

import torch
import pytorch_lightning as pl

if __name__ == '__main__':
    """
    Trains an autoencoder from patches of thermal imaging.

    Command:
        python train.py
    """

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", type=str,
                    default='config/dino_tiny.yml', help="path to the config file")
    args = vars(ap.parse_args())

    
    from src.utils import load_config
    config = load_config(args['config'])

    # prepare output directory
    output_dir = 'trained_models/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # create datamodule
    from src.datamodule import SSLDataModule
    from src.transforms import DINOTransform
    train_transform = DINOTransform(**config["transform"])
    val_transform = DINOTransform(**config["transform"])
    dm = SSLDataModule(data_dir=config["datamodule"]["data_dir"],
                       batch_size=config["datamodule"]["batch_size"],
                       train_transform=train_transform,
                       val_transform=val_transform)

    # define loggers
    from pytorch_lightning.loggers import TensorBoardLogger
    tb_logger = TensorBoardLogger(config["callbacks"]["log_dir"], name=config["experiment"], default_hp_metric=False)

    # define callbacks
    from pytorch_lightning.callbacks import ModelCheckpoint
    mc = ModelCheckpoint(dirpath=os.path.join("trained_models/",config["experiment"]), monitor='val_loss')

    # define lightning module
    from src.lightning_module import TeacherStudentSSLModule

    lm = TeacherStudentSSLModule(img_size=config["transform"]["img_size"],
                                 batch_size=config["datamodule"]["batch_size"],
                                 model_parms=config["model"],
                                 loss_parms=config["loss"],
                                 optimizer_parms=config["optimizer"],
                                 lr_scheduler_parms=config["lr_scheduler"])

    # define trainer
    from pytorch_lightning import Trainer
    trainer = Trainer(gpus=config["trainer"]["devices"],
                      max_epochs=config["trainer"]["max_epochs"],
                      check_val_every_n_epoch=2,
                      limit_val_batches=0.1,
                      precision=16,
                      #gradient_clip_val=0.0, #0 means don’t clip.
                      benchmark=True,
                      logger=tb_logger,
                      callbacks=[mc])

    trainer.fit(lm, dm)

    # manually save trained model
    torch.save(lm.model.student.en, os.path.join(output_dir,"model.pt"))
    # save configuration
    #pl.core.saving.save_hparams_to_yaml(config_yaml=os.path.join(output_dir,'hparams.yaml'), hparams=hparams)
