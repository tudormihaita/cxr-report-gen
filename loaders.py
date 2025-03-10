import torch

from clip import tokenize
from torch.utils.data import DataLoader

class CxrDataLoader(DataLoader):
    def __init__(self, args, split, transform=None):
        self.args = args
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.dataset_name = args.dataset_name
        self.max_seq_length = args.max_seq_length

        self.split = split
        self.transform = transform

        if self.dataset_name == 'iu-xray':
            from datasets import IUXrayDataset
            self.dataset = IUXrayDataset(self.args, split, transform)
        else:
            raise ValueError('Dataset not supported')

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': True if self.split == 'train' else False,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        data = [item for item in data if item is not None]
        if len(data) == 0:
            return None

        uid_batch, text_batch, image_batch, label_batch = zip(*data)

        image_batch = torch.stack(image_batch, 0)
        tokenized_texts_batch = tokenize(list(text_batch), truncate=True)
        mask_batch = (tokenized_texts_batch != 0).long()
        label_batch = torch.stack(label_batch, 0)

        return {
            'uid': uid_batch,
            'text_tokens': tokenized_texts_batch,
            'image': image_batch,
            'attention_mask': mask_batch,
            'medical_concepts': label_batch
        }

