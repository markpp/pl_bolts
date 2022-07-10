import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from networks import Generator, ImageDiscriminator, VideoDiscriminator

class GAN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    	self.G = Generator(config.dim_z, config.dim_a, config.nclasses, config.ch)
        self.ID = ImageDiscriminator(args.ch)
    	self.VD = VideoDiscriminator(args.nclasses, args.ch)

    	# losses
    	self.criterion_gan = nn.BCEWithLogitsLoss()
    	self.criterion_l1 = nn.L1Loss()

    def forward(self, z):
        """Generates an image given input noise z.
        Example::
            z = torch.rand(batch_size, latent_dim)
            gan = GAN.load_from_checkpoint(PATH)
            img = gan(z)
        """
        return self.G(z)

    def configure_optimizers(self):
        optimizer_G = torch.optim.Adam(self.G.parameters(), self.config.g_lr, (0.5, 0.999))
        optimizer_ID = torch.optim.Adam(self.ID.parameters(), self.config.d_lr, (0.5, 0.999))
        optimizer_VD = torch.optim.Adam(self.VD.parameters(), self.config.d_lr, (0.5, 0.999))
        return [optimizer_G, optimizer_ID, optimizer_VD], []


    def training_step(self, batch, batch_idx, optimizer_idx):
        real_vid, _ = batch
        real_start_frame = vid[:,:,0,:,:]
        real_img = real_vid[:,:,random.randint(0, vid.size(2)-1), :, :]

        # train generator
        result = None
        if optimizer_idx == 0:
            result = self.generator_step(x)

            z = torch.randn(bs, args.dim_z).to(device)

            fake_vid = G(real_start_frame, z, cls)
            fake_img = fake_vid[:,:, random.randint(0, vid.size(2)-1),:,:]

            # recon loss
            err_recon = criterion_l1(fake_vid, real_vid)

            # gan loss
            VG_fake = VD(fake_vid, cls)
            IG_fake = ID(fake_img)

            y_real = torch.ones(VG_fake.size()).to(device)
            y_fake = torch.zeros(VG_fake.size()).to(device)

            errVG = criterion_gan(VG_fake, y_real)
            errIG = criterion_gan(IG_fake, y_real)

            # total loss
            errG = args.weight_l1 * err_recon + errVG + errIG

        # train ID
        if optimizer_idx == 1:
            result = self.discriminator_step(x)

            ID_real = ID(real_img)

            ID_fake = ID(fake_img.detach())

            errID = criterion_gan(ID_real, y_real) + criterion_gan(ID_fake, y_fake)

        # train VD
        if optimizer_idx == 2:
            #result = self.discriminator_step(x)

            VD_real = VD(real_vid)
            VD_fake = VD(fake_vid.detach())
            errVD = criterion_gan(VD_real, y_real) + criterion_gan(VD_fake, y_fake)


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

    def save_vids(self, x, x_, name, n=16):
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
