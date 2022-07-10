import argparse
import os, sys
import cv2
import numpy as np
import yaml

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torchsummary import summary
import torch.nn.functional as F


def create_datamodule(config):
    sys.path.append('../../')

    if 'sentinel' in config['exp_params']['data']:
        print("toto")
        '''
        from datamodules.sewer_datamodule import SewerDataModule
        dm = SewerDataModule(data_dir=config['exp_params']['data'],
                             dataset=config['exp_params']['dataset'],
                             batch_size=config['exp_params']['batch_size'],
                             image_size=config['exp_params']['image_size'],
                             in_channels=config['model_params']['in_channels'],
                             n_workers=config['trainer_params']['n_workers'],
                             produce_labels=False)
        dm.setup()
        return dm
        '''
    elif config['exp_params']['dataset']=='themal':
        #from harbour_datamodule import HarbourDataModule
        print("toto")
    else:
        print("no such dataset: {}".format(config['exp_params']['data']))
        return None

if __name__ == '__main__':
    """
    Trains an autoencoder from patches of thermal imaging.

    Command:
        python main.py
    """

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", type=str,
                    default='config/ae.yaml', help="path to the config file")
    ap.add_argument("-p", "--checkpoint", type=str,
                    default='trained_models/model.pt', help="path to the checkpoint file")
    args = vars(ap.parse_args())


    dm = create_datamodule(config)
    if dm == None:
        print("failed to create datamodule")
        exit()

    from lightning_model import GAN
    model = GAN(config)
    # print detailed summary with estimated network size
    #summary(model, (config['model_params']['in_channels'], config['exp_params']['image_size'], config['exp_params']['image_size']), device="cpu")

    from pytorch_lightning.loggers import TensorBoardLogger
    experiment_name = "{}_{}_{}_{}".format(config['logging_params']['name'],
                                           config['exp_params']['dataset'],
                                           config['exp_params']['image_size'],
                                           config['model_params']['in_channels'])

    logger = loggers.TensorBoardLogger(config['logging_params']['save_dir'], name=experiment_name, default_hp_metric=False)

    trainer = Trainer(gpus=config['trainer_params']['gpus'],
                      max_epochs=config['trainer_params']['max_epochs'],
                      #check_val_every_n_epoch=2,
                      #limit_val_batches=0.1,
                      #sync_batchnorm = True, # only necessary for multi-gpu training
                      #precision=16,
                      #gradient_clip_val=0.0, #0 means don’t clip.
                      benchmark=True,
                      logger=logger)


    trainer.fit(model, dm)

    output_dir = 'trained_models/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    torch.save(model.encoder, os.path.join(output_dir,"encoder.pt"))
    torch.save(model.decoder, os.path.join(output_dir,"decoder.pt"))

    #trainer.test(model, dm.test_loader())
