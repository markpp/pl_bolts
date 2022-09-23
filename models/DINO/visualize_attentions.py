import cv2
import timm
import torch
import random
import numpy as np
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from typing import List, Dict
import matplotlib.pyplot as plt
from src.io.io import load_config
from src.transform import ValTransform
from torch.utils.data import DataLoader
from src.model.vit import compute_attentions
from src.model.utils import create_model, load_state_dict_ssl


if __name__ == "__main__":
    
    ckpt_path = "/home/markpp/github/lightning-ssl/checkpoints/epoch=180-step=27511-val_loss=4.682.ckpt"
    config_path = "/home/markpp/github/lightning-ssl/config/ssl/dino_tiny.yml" 

    config = load_config(path=config_path)
    config["model"]

    BACKBONE = config["model"]["backbone"]
    IMG_SIZE = config["transform"]["img_size"]

    model = create_model(
        backbone=BACKBONE,
        pretrained=False,
        img_size=IMG_SIZE
    )

    model = load_state_dict_ssl(
        model=model,
        ssl_state_dict=torch.load(ckpt_path, map_location="cpu")["state_dict"]
    )

    transform = ValTransform(
        model="dino",
        **config["transform"]
    )
    from src.dataset import SSLWEEDS
    dataset = SSLWEEDS(
        root="/home/markpp/datasets/WeedSeason",
        train=False,
        transform=transform
    )

    for _ in range(5):
        i = random.randint(a=0, b=len(dataset)-1)
        img_path = dataset.img_paths[i]
        label = dataset.labels[i]
            
        img = Image.open(img_path)
        img = img.resize((IMG_SIZE, IMG_SIZE))

        # Augmentation
        x, views = transform(img=img)

        # Input Tensor
        x = torch.from_numpy(x).unsqueeze(dim=0)
        print(f"dataset index {i} - path: {dataset.img_paths[i]} - label {label}")

        #plt.imshow(img)
        #plt.show()

        attentions = compute_attentions(
            model=model,
            x=x, 
            patch_size=16
        )

        fig, ax = plt.subplots(figsize=(15, 15), nrows=attentions.shape[0], ncols=3)

        np_img = np.array(img)
        #mask = np.sum(attentions, axis=0)
        for a in range(attentions.shape[0]):
            mask = attentions[a]
            mask = cv2.blur(mask,(10,10))
            mask = np.stack([mask, mask, mask], axis=-1)
            mask = mask / mask.max()
            result = (mask * img).astype("uint8")


            ax[a,0].imshow(img)
            ax[a,0].set_title(f"Original - label {label}")
            ax[a,0].axis("off")

            ax[a,1].imshow(mask)
            ax[a,1].set_title("Attention mask {}".format(a))
            ax[a,1].axis("off")

            ax[a,2].imshow(result)
            ax[a,2].set_title(f"{BACKBONE} - Attention {a} on image")
            ax[a,2].axis("off")

        plt.show()
