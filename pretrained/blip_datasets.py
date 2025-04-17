import os
import torch
import numpy as np
import pandas as pd

from PIL import Image
from utils.minio import MinioUtils
from torch.utils.data import Dataset
from constants import DATASETS_PATH, CHEXPERT_LABELS, MIMIC_IMAGES_PATH


class BlipMimicCXRDataset(Dataset):
    def __init__(self, args, split, processor, max_length=512, use_minio=False):
        self.args = args
        self.use_minio = use_minio
        self.max_length = max_length
        self.processor = processor

        if self.use_minio:
            self.dataset = MinioUtils.load_csv_from_minio(f'mimic-cxr/processed/mimic-cxr_{split}.csv')
        else:
            self.dataset = pd.read_csv(
                os.path.join(str(DATASETS_PATH['mimic-cxr']), 'processed', f'mimic-cxr_{split}.csv'))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        entry = self.dataset.iloc[idx]

        report = str(entry.get('findings', '')).strip() + ' ' + str(entry.get('impression', '')).strip()
        labels_array = entry[CHEXPERT_LABELS].to_numpy(dtype=np.float32)
        labels = torch.tensor(labels_array, dtype=torch.float).squeeze(0)

        if self.use_minio:
            image_path = f'mimic-cxr/{MIMIC_IMAGES_PATH}/{entry["image_path"]}'
            image = MinioUtils.load_image_from_minio(image_path)
        else:
            image_path = os.path.join(str(DATASETS_PATH['mimic-cxr']), MIMIC_IMAGES_PATH, entry['image_path'])
            image = Image.open(image_path).convert('RGB')

        processed = self.processor(
            text=report,
            images=image,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )

        return {
            "input_ids": processed["input_ids"].squeeze(0),
            "pixel_values": processed["pixel_values"].squeeze(0),
            "labels": labels,
        }
