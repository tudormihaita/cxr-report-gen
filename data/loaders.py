import torch
from torch.utils.data import DataLoader

from data import load_transform
from data.datasets import IUXrayDataset, MimicCxrDataset, MimicCxrMVSDataset


class CxrDataLoader(DataLoader):
    def __init__(self, args, split, transform_config=None, tokenizer=None, sampler=None):
        self.args = args
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.drop_last = args.drop_last
        self.use_minio = args.use_minio

        self.image_size = args.image_size
        self.max_length = args.max_length
        self.dataset_name = args.dataset_name

        self.split = split
        self.sampler = sampler
        self.tokenizer = tokenizer
        self.transform = load_transform(split=self.split, transform_config=transform_config)
        self.collate_fn = CxrDataCollator()

        if self.dataset_name == 'iu-xray':
            self.dataset = IUXrayDataset(self.args, split, self.transform)
        elif self.dataset_name == 'mimic-cxr':
            self.dataset = MimicCxrDataset(self.args, split, self.transform, use_minio=self.use_minio)
        elif self.dataset_name == 'mimic-cxr-mvs':
            if self.tokenizer is None:
                raise ValueError('Tokenizer is required for MVS dataset')
            self.collate_fn = CxrMVSDataCollator(self.tokenizer, max_length=self.max_length)
            self.dataset = MimicCxrMVSDataset(self.args, split, self.transform, use_minio=self.use_minio)
        else:
            raise ValueError('Dataset not supported')


        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': self.drop_last,
            'collate_fn': self.collate_fn,
        }

        if self.sampler is not None:
            self.init_kwargs.update({
                'sampler': self.sampler,
                'batch_size': self.batch_size,
                'shuffle': False
            })
        else:
            self.init_kwargs['shuffle'] = (self.split == 'train')

        super().__init__(**self.init_kwargs)


class CxrDataCollator:
    def __call__(self, instances):
        # uid_batch, texts, image_batch, label_batch, caption_batch, idx_batch = zip(*instances)

        image_batch = torch.stack([ins["image"] for ins in instances], dim=0)
        labels_batch = torch.stack([ins["labels"] for ins in instances], dim=0)
        texts_batch = list([ins["text"] for ins in instances])
        captions_batch = list([ins["caption"] for ins in instances])
        uid_batch = list([ins["uid"] for ins in instances])

        idx_batch = list([ins["idx"] for ins in instances])
        idx_batch = torch.stack([torch.tensor(idx, dtype=torch.long) for idx in idx_batch], 0)

        return {
            'uid': uid_batch,
            'idx': idx_batch,
            'image': image_batch,
            'report': texts_batch,
            'labels': labels_batch,
            'caption': captions_batch
        }


class CxrMVSDataCollator:
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, instances):
        images = torch.stack([ins["image"] for ins in instances], dim=0)
        labels = torch.stack([ins["labels"] for ins in instances], dim=0)
        texts = list([ins["text"] for ins in instances])
        text_tokens = self.tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt",
                                     max_length=self.max_length)

        texts2 = list([ins["text2"] for ins in instances])
        text_tokens2 = self.tokenizer(texts2, padding="max_length", truncation=True, return_tensors="pt",
                                      max_length=self.max_length)
        image_views = torch.stack([ins["image_view"] for ins in instances], dim=0)

        batch = {
            "images": images,
            "image_views": image_views,
            "texts": texts,
            "texts2": texts2,
            "text_tokens": text_tokens,
            "text_tokens2": text_tokens2,
            "labels": labels,
        }

        return batch