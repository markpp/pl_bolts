#from torchvision import transforms
import albumentations as Augment

'''
if in_channels == 1:
    mean, std = [0.5], [0.5]
else:
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
'''

def normalize(img, type='standard'):
    if type=='standard':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    if type=='imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    img[0] = (img[0] - mean[0]) / std[0]
    img[1] = (img[1] - mean[1]) / std[1]
    img[2] = (img[2] - mean[2]) / std[2]
    return img

def denormalize(img, type='standard'):
    if type=='standard':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    if type=='imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    img[0] = (img[0] * std[0]) + mean[0]
    img[1] = (img[1] * std[1]) + mean[1]
    img[2] = (img[2] * std[2]) + mean[2]
    return img


def Normalize(type='standard'):
    if type=='standard':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    if type=='imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    return Augment.Normalize(mean, std)

def Denormalize(type='standard'):
    if type=='standard':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    if type=='imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    return Augment.Normalize((-mean / std), (1.0 / std))
