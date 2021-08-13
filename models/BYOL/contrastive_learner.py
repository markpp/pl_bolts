import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision import models, transforms
from torch.optim import Adam
from torchsummary import summary
import numpy as np

from byol_pytorch import BYOL

class ContrastiveLearner(pl.LightningModule):
    def __init__(self, net, config):
        super().__init__()
        self.config = config
        self.BYOL = BYOL(net=net,
                         image_size=self.config['exp_params']['image_size'],
                         hidden_layer=self.config['model_params']['hidden_layer'],
                         projection_size=self.config['model_params']['projection_size'],
                         projection_hidden_size=self.config['model_params']['projection_hidden_size'],
                         moving_average_decay=self.config['model_params']['moving_average_decay'],
                         use_momentum=True)

    def forward(self, x):
        return self.BYOL(x)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.config['exp_params']['LR'], betas=(self.config['exp_params']['beta1'], self.config['exp_params']['beta2']))

    def on_before_zero_grad(self, _):
        if self.BYOL.use_momentum:
            self.BYOL.update_moving_average()

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log('loss', loss, on_step=True, prog_bar=False)
        return {"loss": loss}

    '''
    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        # save input and output images at beginning of epoch
        if batch_idx == 0:
            self.save_images(x, output, "val_input_output")
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss, on_epoch=True, prog_bar=True)

    def save_images(self, x, output, name, n=16):
        """
        Saves a plot of n images from input and output batch
        """
        # make grids and save to logger
        grid_top = vutils.make_grid(x[:n,:,:,:], nrow=n)
        grid_bottom = vutils.make_grid(output[:n,:,:,:], nrow=n)
        grid = torch.cat((grid_top, grid_bottom), 1)
        self.logger.experiment.add_image(name, grid, self.current_epoch)
    '''
