import torch

from torch.utils.data import DataLoader
from datasets import IUXrayDataset, MimicCXRDataset

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


def build_transform(split, image_size=224, min_scale=0.5):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    def _convert_image_to_rgb(image):
        return image.convert("RGB")

    # TODO: add randaugment based on BLIP implementation
    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(min_scale, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            normalize
        ])


class CxrDataLoader(DataLoader):
    def __init__(self, args, split, transform=None, sampler=None):
        self.args = args
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.dataset_name = args.dataset_name
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

        uid_batch, report_batch, image_batch, label_batch = zip(*data)

        image_batch = torch.stack(image_batch, 0)
        label_batch = torch.stack(label_batch, 0)

        return {
            'uid': uid_batch,
            'image': image_batch,
            'report': report_batch,
            'labels': label_batch
        }

