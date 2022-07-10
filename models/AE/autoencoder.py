import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.optim import Adam
from torchsummary import summary
import numpy as np

from model import create_encoder, create_decoder

class Autoencoder(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = create_encoder(config)
        self.decoder = create_decoder(config)

    def forward(self, x):
        z = self.encoder(x)
        x_ = self.decoder(z)
        return x_

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.config['exp_params']['LR'], betas=(self.config['exp_params']['beta1'], self.config['exp_params']['beta2']))

    def training_step(self, batch, batch_idx):
        x = batch
        output = self(x)
        loss = F.mse_loss(output, x)
        self.log('loss', loss, on_step=True, prog_bar=False)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x = batch
        output = self(x)
        loss = F.mse_loss(output, x)

        # save input and output images at beginning of epoch
        if batch_idx == 0:
            self.save_images(x, output, "val_input_output")

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x = batch
        x_ = self(x)
        loss = F.mse_loss(x, x_)

        # save input and output images at beginning of epoch
        if batch_idx == 0:
            self.save_images(x, x_, "test_input_vs_reconstruction")

        logs = {"test_loss": loss}
        return {"test_loss": loss, "log": logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        self.log('avg_test_loss', avg_loss, on_epoch=True)

    def save_images(self, x, x_, name, n=16):
        """
        Saves a plot of n images from input and output batch
        """
        # make grids and save to logger
        #grid_top = vutils.make_grid(x[:n,:,:,:], nrow=n)
        #grid_bottom = vutils.make_grid(x_[:n,:,:,:], nrow=n)
        grid_top = vutils.make_grid(x[:n,:,:,:], nrow=n, normalize=True, value_range=(-1,1))
        grid_bottom = vutils.make_grid(x_[:n,:,:,:], nrow=n, normalize=True, value_range=(-1,1))
        grid = torch.cat((grid_top, grid_bottom), 1)
        self.logger.experiment.add_image(name, grid, self.current_epoch)
