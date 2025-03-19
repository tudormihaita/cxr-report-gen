import os
import torch
import pandas as pd
import numpy as np

from PIL import Image
from torch.utils.data import Dataset

ROOT_PATH = '/Volumes/T7 Shield/datasets/iu-xray'
IMAGES_PATH = os.path.join(ROOT_PATH, 'images/images_normalized')
LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']

class IUXrayDataset(Dataset):
    def __init__(self, args, split_path, transform=None):
        self.args = args
        self.dataset = pd.read_csv(os.path.join(ROOT_PATH, "processed", split_path))
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        entry = self.dataset.iloc[idx]
        uid = entry['uid']

        # text = str(entry['impression']) + ' ' + str(entry['findings'])
        text = str(entry['report'])
        labels_array = entry[LABELS].to_numpy(dtype=np.float32)
        labels = torch.tensor(labels_array, dtype=torch.float).squeeze(0)
        labels = torch.where(labels == -1, torch.tensor(0.0), labels)


        proj_path = entry['frontal_filename'] if pd.notna(entry['frontal_filename']) else entry['lateral_filename']
        image = Image.open(os.path.join(IMAGES_PATH, proj_path)).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return uid, text, image, labels