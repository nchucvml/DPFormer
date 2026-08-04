from continuum import ClassIncremental
from continuum.datasets import CIFAR100, ImageNet100, ImageFolderDataset
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from torchvision.transforms import functional
import copy
import os
import numpy as np

try:
    interpolation = functional.InterpolationMode.BICUBIC
except AttributeError:
    interpolation = 3


class ImageNet1000(ImageFolderDataset):
    """Continuum dataset for datasets with tree-like folder structure.
    :param data_path: Root folder containing 'train' and 'val' subdirectories.
    :param train: If True, loads the training split; otherwise loads validation.
    :param download: Unused, kept for interface compatibility.
    """

    def __init__(
            self,
            data_path: str,
            train: bool = True,
            download: bool = False,
    ):
        super().__init__(data_path=data_path, train=train, download=download)

    def get_data(self):
        if self.train:
            self.data_path = os.path.join(self.data_path, "train")
        else:
            self.data_path = os.path.join(self.data_path, "val")
        return super().get_data()


def build_transform(is_train, args):
    augmentation = None if args.augmentation == 'none' else args.augmentation
    resize_img = args.input_size > 32
    if is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            auto_augment=augmentation,
            interpolation='bicubic',
        )
        if args.dataset.lower() == 'cifar100':
            # replace RandomResizedCropAndInterpolation with RandomCrop
            if args.pretrained_model is None:
                transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
            transform.transforms[-1] = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

        return transform
    # for test
    t = []
    if resize_img:
        t.append(transforms.Resize(256, interpolation=interpolation))    # standard ImageNet eval: resize short side to 256, then center crop to input_size
        t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    if args.dataset.lower() == 'cifar100':
        t.append(transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)))
    else:
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

    return transforms.Compose(t)


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.dataset.lower() == 'cifar100':
        dataset = CIFAR100(args.data_path, train=is_train, download=True)
    elif args.dataset.lower() == 'imagenet100':
        dataset = ImageNet100(
            args.data_path, train=is_train,
            data_subset=os.path.join('./datasets/imagenet100_splits', "train_100.txt" if is_train else "val_100.txt")
        )
    elif args.dataset.lower() == 'imagenet1000' or args.dataset.lower() == 'imagenetr':
        dataset = ImageNet1000(args.data_path, train=is_train)
    else:
        raise ValueError(f'Unknown dataset {args.dataset}.')

    scenario = ClassIncremental(
        dataset,
        initial_increment=args.increment,
        increment=args.increment,
        transformations=transform.transforms,
        class_order=args.class_order
    )

    return scenario


def get_finetuning_dataset(dataset, memory, finetuning='balanced', oversample_old=1, task_id=0):
    if finetuning == 'balanced':
        x, y, t = memory.get()

        if oversample_old > 1 and task_id > 0:
            old_indexes = np.where(t < task_id)[0]
            assert len(old_indexes) > 0
            new_indexes = np.where(t >= task_id)[0]

            indexes = np.concatenate([
                np.repeat(old_indexes, oversample_old),
                new_indexes
            ])
            x, y, t = x[indexes], y[indexes], t[indexes]

        new_dataset = copy.deepcopy(dataset)
        new_dataset._x = x
        new_dataset._y = y
        new_dataset._t = t
    elif finetuning in ('all', 'none'):
        new_dataset = dataset
    else:
        raise NotImplementedError(f'Unknown finetuning method {finetuning}')

    return new_dataset
