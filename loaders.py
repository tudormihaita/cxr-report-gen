import torch

from torch.utils.data import DataLoader
from datasets import IUXrayDataset

class CxrDataLoader(DataLoader):
    def __init__(self, args, split, transform=None, sampler=None):
        self.args = args
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.dataset_name = args.dataset_name
        self.max_seq_length = args.max_seq_length

        self.split = split
        self.sampler = sampler
        self.transform = transform

        if self.dataset_name == 'iu-xray':
            split_path = f'iu_xray_{self.split}.csv'
            self.dataset = IUXrayDataset(self.args, split_path, transform)
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

        uid_batch, text_batch, image_batch, label_batch = zip(*data)

        image_batch = torch.stack(image_batch, 0)
        label_batch = torch.stack(label_batch, 0)

        # padding mask used in conditional generation input to prevent attending to padded text tokens
        # TODO: move mask computation and tokenization outside batch
        # mask_batch = (tokenized_texts_batch != 0).long()

        return {
            'uid': uid_batch,
            'image': image_batch,
            'report': text_batch,
            'labels': label_batch
        }

