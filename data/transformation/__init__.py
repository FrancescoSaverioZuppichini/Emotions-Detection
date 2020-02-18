import numpy as np
import torchvision.transforms as T
from imgaug import augmenters as iaa


class ImgAugTransform:
    """
    Wrapper to allow imgaug to work with Pytorch transformation pipeline
    """
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.8,
                          iaa.Sequential([iaa.Fliplr(0.5),
                                          iaa.Flipud(0.5)])),
            # iaa.Sometimes(0.2, iaa.Crop(percent=(0, 0.1))),
            # iaa.Sometimes(0.2, iaa.Affine(rotate=(-20, 20), mode='symmetric'))
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.aug.augment_image(img)
        return img


val_transform = T.Compose([T.Grayscale(), T.ToTensor()])
train_transform = T.Compose([ImgAugTransform(), T.ToPILImage(), T.Grayscale(), T.ToTensor()])
