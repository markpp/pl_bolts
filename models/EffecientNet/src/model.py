import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import os, sys
import pytorch_lightning as pl
import cv2

from efficientnet_pytorch import EfficientNet

from transforms.normalization import denormalize

def plot_prediction(imgs, targets, preds):
    """
    Plot the target and predictions
    """
    dets = []
    for img, tar, pred in zip(imgs, targets, preds):
        out = denormalize(img.cpu()).permute(1, 2, 0).mul(255).byte().numpy()
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.putText(out,"target: {:.2f}".format(tar.cpu().numpy()[0]), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.putText(out,"pred: {:.2f}".format(pred.cpu().numpy()[0]), (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

        if len(dets):
            dets = np.concatenate((dets, out), axis=1)
        else:
            dets = out
    return dets

class Classifier(pl.LightningModule):

    def __init__(self, hparams, pretrained=True):
      super().__init__()

      self.lr = hparams.learning_rate
      # init model
      if pretrained:
          self.model = EfficientNet.from_pretrained(hparams.arch)
      else:
          self.model = EfficientNet.from_name(hparams.arch)

      self.model._fc = nn.Linear(self.model._fc.in_features, hparams.output_size)

      self.criterion = nn.MSELoss()
      self.eval_criterion = nn.L1Loss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        input, target = batch
        output = self.forward(input)
        loss = self.criterion(output, target)
        self.log('loss', loss, on_step=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        input, target = batch
        output = self.forward(input)

        if batch_idx == 0:
            res = plot_prediction(input, target, output)
            self.logger.experiment.add_image("val images", res, self.current_epoch, dataformats='HWC')

        loss = self.criterion(output, target)
        diff = self.eval_criterion(output, target)
        return {'loss': loss, 'diff': diff}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        diff = torch.stack([x['diff'] for x in outputs]).mean()
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_diff', diff, on_epoch=True)

    def test_step(self, batch, batch_idx):
        input, target = batch
        output = self.forward(input)

        if batch_idx == 0:
            res = plot_prediction(input, target, output)
            self.logger.experiment.add_image("test images", res, self.current_epoch, dataformats='HWC')

        loss = self.criterion(output, target)
        diff = self.eval_criterion(output, target)
        return {'loss': loss, 'diff': diff}

    def test_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        diff = torch.stack([x['diff'] for x in outputs]).mean()
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_diff', diff, on_epoch=True)
