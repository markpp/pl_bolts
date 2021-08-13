#from torchvision import transforms

import albumentations as Augment
import cv2

from transforms.normalization import Normalize

def basic_augmentation(img_height, img_width, img_pad=0, in_channels=3):
    return Augment.Compose([Augment.Resize(img_height+img_pad, img_width+img_pad, interpolation=cv2.INTER_NEAREST, always_apply=True),
                            Augment.RandomCrop(img_height, img_width, always_apply=True),
                            Augment.HorizontalFlip(p=0.5),
                            Augment.RandomBrightnessContrast(p=1.0),
                            Normalize()
                            ])
