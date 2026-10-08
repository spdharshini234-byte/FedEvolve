"""Image transforms. No colour jitter on purpose: 'color defect' depends on colour."""
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transforms(image_size: int, train: bool = False, augment: bool = True):
    steps = [transforms.Resize((image_size, image_size))]
    if train and augment:
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                  transforms.RandomAffine(degrees=10, translate=(0.05, 0.05))]
    steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(steps)
