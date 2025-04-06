import os
import torch
import pandas as pd
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

from constants import DATASETS_PATH, IMAGES_PATH, CHEXPERT_LABELS


def build_iu_xray_sampler(split, label_columns=CHEXPERT_LABELS):
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

        report_text = str(entry['report'])
        prompt_text = str(entry['prompt'])
        labels_array = entry[CHEXPERT_LABELS].to_numpy(dtype=np.float32)
        labels = torch.tensor(labels_array, dtype=torch.float).squeeze(0)

        # TODO: load both frontal and lateral projections if present to experiment consistency
        proj_path = entry['frontal_filename'] if pd.notna(entry['frontal_filename']) else entry['lateral_filename']
        image = Image.open(os.path.join(IMAGES_PATH['iu-xray'], proj_path)).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return uid, report_text, prompt_text, image, labels