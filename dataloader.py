from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageEnhance
import torch
import numpy as np

transformtypedict = dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast,
                         Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)

class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))
        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (randtensor[i] * 2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')
        return out


class TransformLoader:
    def __init__(self, image_size,
                 normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param

    def parse_transform(self, transform_type):
        if transform_type == 'ImageJitter':
            method = ImageJitter(self.jitter_param)
            return method
        method = getattr(transforms, transform_type)

        if transform_type == 'RandomResizedCrop':
            return method(self.image_size)
        elif transform_type == 'CenterCrop':
            return method(self.image_size)
        elif transform_type == 'Resize':
            return method([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        elif transform_type == 'Normalize':
            return method(**self.normalize_param)
        else:
            return method()

    def get_composed_transform(self, aug=False):
        if aug:
            transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Resize', 'CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform


def get_transform(phase, image_size):
    trans_loader = TransformLoader(image_size)
    if phase == "train":
        transform = trans_loader.get_composed_transform(aug=True)

    elif phase == "test":
        transform = trans_loader.get_composed_transform(aug=False)
    else:
        print("unknow phase")
        exit()
    return transform


class BaseDataset(Dataset):

    def __init__(self, phase, image_size, label2id):
        self.data = []
        self.targets = []
        self.label2id = label2id
        self.transform = get_transform(phase, image_size)

    def __getitem__(self, index):
        path, label = self.data[index], self.targets[index]
        image = self.transform(Image.open(path).convert('RGB'))
        label = int(label)

        return image, label

    def __len__(self):
        return len(self.data)


class BaseDataset_flip(Dataset):

    def __init__(self, phase, image_size, label2id):
        self.data = []
        self.targets = []
        self.label2id = label2id
        self.transform = get_transform(phase, image_size)

    def __getitem__(self, index):
        path, label = self.data[index], self.targets[index]
        image = self.transform(Image.open(path).convert('RGB'))
        image = torch.flip(image, [2])
        label = int(label)

        return image, label

    def __len__(self):
        return len(self.data)


class BaseDataset_flip_flag(Dataset):

    def __init__(self, phase, image_size, label2id):
        self.data = []
        self.targets = []
        self.flags = []
        self.on_flags = []
        self.label2id = label2id
        self.transform = get_transform(phase, image_size)

    def __getitem__(self, index):
        path, label = self.data[index], self.targets[index]
        flags = self.flags[index]
        on_flags = self.on_flags[index]
        image = self.transform(Image.open(path).convert('RGB'))
        image = torch.flip(image, [2])
        label = int(label)

        return image, label, flags, on_flags

    def __len__(self):
        return len(self.data)



class BaseDataset_flag(Dataset):

    def __init__(self, phase, image_size, label2id):
        self.data = []
        self.targets = []
        self.flags = []
        self.on_flags = []
        self.label2id = label2id
        self.transform = get_transform(phase, image_size)

    def __getitem__(self, index):
        path, label = self.data[index], self.targets[index]
        flags = self.flags[index]
        on_flags = self.on_flags[index]
        image = self.transform(Image.open(path).convert('RGB'))
        label = int(label)

        return image, label, flags, on_flags

    def __len__(self):
        return len(self.data)

class BaseDataset_ul_flag(Dataset):

    def __init__(self, phase, image_size, label2id):
        self.data = []
        self.targets = []
        self.flags = []
        self.label2id = label2id
        self.transform = get_transform(phase, image_size)

    def __getitem__(self, index):
        path, label = self.data[index], self.targets[index]
        flags = self.flags[index]
        image = self.transform(Image.open(path).convert('RGB'))
        label = label

        return image, label, flags

    def __len__(self):
        return len(self.data)

class UnlabelDataset(Dataset):

    def __init__(self, image_size, unlabeled_num=None):
        self.data = []
        self.transform = get_transform("train", image_size)

        if unlabeled_num != -1 and unlabeled_num is not None:
            try:
                self.data = self.data[: unlabeled_num]
            except:
                pass

    def __getitem__(self, index):
        path = self.data[index]
        image = self.transform(Image.open(path).convert('RGB'))

        return image

    def __len__(self):
        return len(self.data)



