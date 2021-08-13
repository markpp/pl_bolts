from torchvision import transforms

def largest_center_crop(img):
    h, w, c = img.shape
    if h > w:
        top_h = int((h - w) / 2)
        img = img[top_h:top_h + w]
    else:
        left_w = int((w - h) / 2)
        img = img[:, left_w:left_w + h]
    return img


'''
import albumentations as Augment

test_t = Augment.Compose([Augment.SmallestMaxSize(max_size=hparams.image_height, interpolation=cv2.INTER_LINEAR, always_apply=True),
                          Augment.CenterCrop(hparams.image_height, hparams.image_height, always_apply=True),
                          Augment.Normalize(mean, std)
              ])
'''
