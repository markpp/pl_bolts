import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import List, Tuple, Dict
from torch.optim import Optimizer, Adam, SGD, AdamW

def cancel_gradients_last_layer(epoch, model, frozen_epochs):
    if epoch >= frozen_epochs:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None

class TeacherStudentSSLModule(pl.LightningModule):
    
    def __init__(
        self, 
        img_size: int,
        batch_size: int,
        model_parms: Dict,
        loss_parms: Dict,
        optimizer_parms: Dict,
        lr_scheduler_parms: Dict,
        last_layer_frozen: int = 2,
    ) -> None:
        
        super().__init__()     

        # setting up model, loss, optimizer, lr_scheduler
        from src.dino import DINO
        self.model = DINO(img_size=img_size, **model_parms)

        from src.utils import DINOLoss
        self.criterion = DINOLoss(**loss_parms)
        
        # adapt lr to rule (base_lr * batch_size / 256)
        optimizer_parms["lr"] *= batch_size / 256.
        self.optimizer = AdamW(params=self.model.parameters(), **optimizer_parms)

        #if lr_scheduler_parms["name"] == "cosine":
        self.lr_scheduler = CosineAnnealingWarmRestarts(optimizer=self.optimizer, **lr_scheduler_parms)
        #if lr_scheduler_parms["name"] == "linear_cosine":
            #self.lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer=self.optimizer, **lr_scheduler_parms)

        self.last_layer_frozen = last_layer_frozen
        
    def forward(self, views):
        return self.model(views)
        
    def training_step(self, batch, batch_idx):
        
        x, views, _ = batch
        outputs = self(views)
        loss = self.criterion(outputs)
        
        cancel_gradients_last_layer(
            epoch=self.current_epoch,
            model=self.model,
            frozen_epochs=self.last_layer_frozen
        )
        # EMA update        
        self.model.update_teacher()
        
        self.log("loss/train", loss, sync_dist=True, prog_bar=True)
        self.log("lr", self.lr_scheduler.get_last_lr()[0], prog_bar=True)        
        return loss
    
    def validation_step(self, batch, batch_idx):  
        x, views, targets = batch
        outputs = self(views)
        loss = self.criterion(outputs)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log("loss/val", avg_loss, sync_dist=True, prog_bar=True)
        
    def configure_optimizers(self):
        if self.lr_scheduler is None:
            return [self.optimizer]
        else:
            return [self.optimizer], [self.lr_scheduler]