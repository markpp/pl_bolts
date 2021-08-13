import argparse
import os, sys
import cv2
import numpy as np
import yaml

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers


def create_datamodule(config):
    sys.path.append('../../../datasets')

    if 'sewer' in config['exp_params']['dataset']:
        from sewer.datamodule import SewerDataModule
        dm = SewerDataModule(data_dir=config['exp_params']['data'],
                             batch_size=config['exp_params']['batch_size'],
                             image_size=config['exp_params']['image_size'],
                             n_workers=config['trainer_params']['n_workers'])
        dm.setup()
        return dm
    elif 'themal' in config['exp_params']['dataset']:
        from harbour_datamodule import HarbourDataModule
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
                    default='config/byol.yaml', help="path to the config file")
    ap.add_argument("-p", "--checkpoint", type=str,
                    default='trained_models/model.pt', help="path to the checkpoint file")
    args = vars(ap.parse_args())

    with open(args['config'], 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    dm = create_datamodule(config)
    if dm == None:
        print("failed to create datamodule")
        exit()

    from torchvision import models
    resnet = models.resnet50(pretrained=True)

    experiment_name = "{}_{}_{}_{}".format(config['logging_params']['name'],
                                           config['exp_params']['dataset'],
                                           config['exp_params']['image_size'],
                                           config['model_params']['in_channels'])

    logger = loggers.TensorBoardLogger(config['logging_params']['save_dir'], name=experiment_name)

    from contrastive_learner import ContrastiveLearner
    learner = ContrastiveLearner(resnet, config)

    trainer = Trainer(gpus=config['trainer_params']['gpus'],
                      max_epochs=config['trainer_params']['max_epochs'],
                      sync_batchnorm = True, # only necessary for multi-gpu training
                      precision=16,
                      benchmark=True,
                      logger=logger)

    trainer.fit(learner, dm)

    output_dir = 'trained_models/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    torch.save(learner.BYOL, os.path.join(output_dir,"model.pt"))
    #torch.save(learner, os.path.join(output_dir,"learner.pt"))
