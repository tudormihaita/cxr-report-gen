import torch
import random

from torch.utils.data import DataLoader, Subset
from data.datasets import IUXrayDataset, MimicCXRDataset

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


class TransformBuilder:
    def __init__(self, image_size=224, min_scale=0.8, normalize_stats=None, interpolation=InterpolationMode.BICUBIC):
        self.image_size = image_size
        self.min_scale = min_scale
        self.normalize_stats = normalize_stats or ((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944))
        self.interpolation = interpolation

    @staticmethod
    def _convert_image_to_rgb(image):
        return image.convert("RGB")

    def build(self, split):
        normalize = transforms.Normalize(self.normalize_stats[0], self.normalize_stats[1])

        if split == 'train':
            return transforms.Compose([
                transforms.Resize(self.image_size, interpolation=self.interpolation),
                transforms.RandomResizedCrop(
                    self.image_size,
                    scale=(self.min_scale, 1.0),
                    ratio=(0.9, 1.1),
                    interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomAffine(
                    degrees=5,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                ),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                self._convert_image_to_rgb,
                transforms.ToTensor(),
                normalize
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.image_size, interpolation=InterpolationMode.BICUBIC),
                self._convert_image_to_rgb,
                transforms.ToTensor(),
                normalize
            ])


class CxrDataLoader(DataLoader):
    def __init__(self, args, split, transform=None, sampler=None):
        self.args = args
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.dataset_name = args.dataset_name
        self.drop_last = args.drop_last
        self.use_minio = args.use_minio

        self.split = split
        self.sampler = sampler
        self.transform = transform

        if self.dataset_name == 'iu-xray':
            split_path = f'iu_xray_{self.split}.csv'
            self.dataset = IUXrayDataset(self.args, split_path, transform)

        elif self.dataset_name == 'mimic-cxr':
            self.dataset = MimicCXRDataset(self.args, split, transform, use_minio=self.use_minio)
        else:
            raise ValueError('Dataset not supported')

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers,
            'drop_last': self.drop_last
        }

        if self.sampler is not None:
            self.init_kwargs['sampler'] = self.sampler
            self.init_kwargs['shuffle'] = False
            self.init_kwargs['batch_size'] = self.batch_size
        else:
            self.init_kwargs['shuffle'] = (self.split == 'train')

        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        data = [item for item in data if item is not None]
        if len(data) == 0:
            return None

        uid_batch, report_batch, image_batch, label_batch, caption_batch, idx_batch = zip(*data)

        idx_batch = torch.stack([torch.tensor(idx, dtype=torch.long) for idx in idx_batch], 0)
        image_batch = torch.stack(image_batch, 0)
        label_batch = torch.stack(label_batch, 0)

        return {
            'uid': uid_batch,
            'idx': idx_batch,
            'image': image_batch,
            'report': report_batch,
            'labels': label_batch,
            'caption': caption_batch
        }
