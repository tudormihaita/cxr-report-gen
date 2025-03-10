import os
import torch
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset

BASE_PATH = '/Volumes/T7 Shield/datasets/iu-xray'
LABELS_PATH = os.path.join(BASE_PATH, 'indiana_labels.csv')
REPORTS_PATH = os.path.join(BASE_PATH, 'indiana_reports.csv')
IMAGES_PATH = os.path.join(BASE_PATH, 'images/images_normalized')
PROJECTIONS_PATH = os.path.join(BASE_PATH, 'indiana_projections.csv')

class IUXrayDataset(Dataset):
    def __init__(self, args, split, transform=None):
        self.reports = pd.read_csv(REPORTS_PATH)
        self.labels = pd.read_csv(LABELS_PATH)
        self.projections = pd.read_csv(PROJECTIONS_PATH)

        self.split = split
        self.transform = transform

        self.labels.fillna(0, inplace=True)

    def __len__(self):
        return len(self.reports)

    def __getitem__(self, idx):
        report = self.reports.iloc[idx]
        uid = report['uid']

        text = str(report['impression']) + ' ' + str(report['findings'])

        labels = self.labels[(self.labels['uid'] == uid)]
        if labels.empty:
            values = torch.zeros(14, dtype=torch.float)
        else:
            values = torch.tensor(labels.iloc[:, 2:].values, dtype=torch.float).squeeze(0)

        proj_path = self.projections[(self.projections['uid'] == uid) & (self.projections['projection'] == 'Frontal')]

        if proj_path.empty:
            return None

        image = Image.open(os.path.join(IMAGES_PATH, proj_path.iloc[0]['filename'])).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return uid, text, image, values
