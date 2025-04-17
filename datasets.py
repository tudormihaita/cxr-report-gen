import os
import torch
import pandas as pd
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms import ToTensor

from utils.minio import MinioUtils
from constants import DATASETS_PATH, CHEXPERT_LABELS, IU_XRAY_IMAGES_PATH, MIMIC_IMAGES_PATH


def _build_iu_xray_sampler(split, label_columns=CHEXPERT_LABELS):
    """
    Create a weighted sampler for the IU-Xray dataset based on the labels.
    :param split: the split of the dataset (train, val, test)
    :param label_columns: list of columns containing the labels
    :return: WeightedRandomSampler
    """
    df = pd.read_csv(os.path.join(str(DATASETS_PATH['iu-xray']), 'processed', f'iu_xray_{split}.csv'))
    labels_array = df[label_columns].replace(-1, 0).values
    label_freq = np.sum(labels_array, axis=0)
    inverse_freq = 1.0 / label_freq
    sample_weights = np.max(labels_array * inverse_freq, axis=1)
    return WeightedRandomSampler(torch.DoubleTensor(sample_weights), num_samples=len(df), replacement=True)


class IUXrayDataset(Dataset):
    def __init__(self, args, split_path, transform=None):
        self.args = args
        self.dataset = pd.read_csv(os.path.join(str(DATASETS_PATH['iu-xray']), 'processed', split_path))
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        entry = self.dataset.iloc[idx]
        uid = entry['uid']

        report = str(entry['impression'])
        labels_array = entry[CHEXPERT_LABELS].to_numpy(dtype=np.float32)
        labels = torch.tensor(labels_array, dtype=torch.float).squeeze(0)

        # TODO: load both frontal and lateral projections if present to experiment consistency
        proj_path = entry['frontal_filename'] if pd.notna(entry['frontal_filename']) else entry['lateral_filename']
        image = Image.open(os.path.join(str(DATASETS_PATH['iu-xray']), IU_XRAY_IMAGES_PATH, proj_path)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        else:
            image = ToTensor()(image)

        return uid, report, image, labels


class MimicCXRDataset(Dataset):
    def __init__(self, args, split, transform=None, use_minio=False):
        self.args = args
        self.use_minio = use_minio
        self.transform = transform

        if self.use_minio:
            self.dataset = MinioUtils.load_csv_from_minio(f'mimic-cxr/processed/mimic-cxr_{split}.csv')
        else:
            self.dataset = pd.read_csv(
                os.path.join(str(DATASETS_PATH['mimic-cxr']), 'processed', f'mimic-cxr_{split}.csv'))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        entry = self.dataset.iloc[idx]
        uid = entry['dicom_id']

        report = str(entry.get('findings', '')).strip() + ' ' + str(entry.get('impression', '')).strip()
        labels_array = entry[CHEXPERT_LABELS].to_numpy(dtype=np.float32)
        labels = torch.tensor(labels_array, dtype=torch.float).squeeze(0)

        if self.use_minio:
            image_path = f'mimic-cxr/{MIMIC_IMAGES_PATH}/{entry["image_path"]}'
            image = MinioUtils.load_image_from_minio(image_path)
        else:
            image_path = os.path.join(str(DATASETS_PATH['mimic-cxr']), MIMIC_IMAGES_PATH, entry['image_path'])
            image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        else:
            image = ToTensor()(image)

        return uid, report, image, labels
