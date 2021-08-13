import os, sys
import numpy as np

import torch
import pytorch_lightning as pl

from config import hparams

if __name__ == '__main__':
    """
    Command:
        python train.py
    """

    sys.path.append('../../')
    from datamodules.sewer_datamodule import SewerDataModule

    dm = SewerDataModule(data_dir=hparams.data,
                         dataset=hparams.dataset,
                         batch_size=hparams.batch_size,
                         image_size=hparams.image_size,
                         in_channels=hparams.in_channels,
                         n_workers=hparams.n_workers)
    dm.setup()
    print("sample shape {}".format(dm.data_val[0][0].shape))

    from src.model import Classifier
    model = Classifier(hparams)

    from pytorch_lightning.loggers import TensorBoardLogger
    experiment_name = "{}_{}_{}_{}".format(hparams.name,
                                           hparams.dataset,
                                           hparams.image_size,
                                           hparams.in_channels)
    logger = TensorBoardLogger(hparams.save_dir, name=experiment_name, default_hp_metric=False)

    from pytorch_lightning import Trainer
    trainer = Trainer(gpus=hparams.gpus,
                      max_epochs=hparams.max_epochs,
                      #check_val_every_n_epoch=2,
                      limit_val_batches=0.1,
                      sync_batchnorm = True, # only necessary for multi-gpu training
                      precision=16,
                      gradient_clip_val=0.0, #0 means don’t clip.
                      benchmark=True,
                      logger=logger)

    trainer.fit(model, dm)

    # save model
    output_dir = 'trained_models/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    torch.save(model.model.state_dict(), os.path.join(output_dir,'model_dict.pth'))
    torch.save(model.model, os.path.join(output_dir,'model.pt'))
