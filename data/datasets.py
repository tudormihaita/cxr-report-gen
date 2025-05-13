import os
import nltk
import torch
import random
import pandas as pd
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from nltk.tokenize import sent_tokenize

from utils.minio import MinioUtils
from data import generate_image_caption, build_image_text_similarity_mappings, transform_image
from constants import DATASETS_PATH, CHEXPERT_LABELS, IU_XRAY_IMAGES_PATH, MIMIC_IMAGES_PATH


class IUXrayDataset(Dataset):
    def __init__(self, args, split, transform):
        self.args = args
        self.split = split
        self.dataset = pd.read_csv(os.path.join(str(DATASETS_PATH['iu-xray']), 'processed', f'iu_xray_{split}.csv'))
        self.transform = transform
        self.uid2idx = {uid: idx for idx, uid in enumerate(set(self.dataset['uid'].astype(str)))}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        entry = self.dataset.iloc[idx]
        uid = entry['uid']
        idx = self.uid2idx[uid]

        report = str(entry['impression'])
        labels_array = (entry[CHEXPERT_LABELS].replace(-1, 0)).to_numpy(dtype=np.float32)
        labels = torch.tensor(labels_array, dtype=torch.float).squeeze(0)
        caption = generate_image_caption(labels_array)

        # TODO: load both frontal and lateral projections if present
        proj_path = entry['frontal_filename'] if pd.notna(entry['frontal_filename']) else entry['lateral_filename']
        image = Image.open(os.path.join(str(DATASETS_PATH['iu-xray']), IU_XRAY_IMAGES_PATH, proj_path)).convert('RGB')
        image = transform_image(self.transform, image)

        return {
            'uid': uid,
            'idx': idx,
            'image': image,
            'text': report,
            'labels': labels,
            'caption': caption
        }


class MimicCxrDataset(Dataset):
    def __init__(self, args, split, transform, use_minio=False):
        self.args = args
        self.split = split
        self.use_minio = use_minio
        self.transform = transform

        if self.use_minio:
            self.dataset = MinioUtils.load_csv_from_minio(f'mimic-cxr/processed/mimic-cxr_{split}.csv')
        else:
            self.dataset = pd.read_csv(
                os.path.join(str(DATASETS_PATH['mimic-cxr']), 'processed-cxr-pro', f'mimic-cxr_{split}.csv'))

        uids = (self.dataset['dicom_id']).astype(str).tolist()
        reports = (self.dataset['report'].fillna('')).astype(str).tolist()
        # captions = [generate_image_caption(labels) for labels in (self.dataset[CHEXPERT_LABELS].replace(-1, 0)).to_numpy()]

        self.uid2idx = {uid: idx for idx, uid in enumerate(set(self.dataset['dicom_id'].astype(str)))}
        self.img2txt = build_image_text_similarity_mappings(reports, uids)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        entry = self.dataset.iloc[idx]
        uid = entry['dicom_id']
        idx = self.uid2idx[uid]

        # report = str(entry.get('findings', '')).strip() + ' ' + str(entry.get('impression', '')).strip()
        report = str(entry.get('report', '')).strip()
        labels_array = (entry[CHEXPERT_LABELS].replace(-1, 0)).to_numpy(dtype=np.float32)
        labels = torch.tensor(labels_array, dtype=torch.float).squeeze(0)
        caption = generate_image_caption(labels_array)
        if self.use_minio:
            image_path = f'mimic-cxr/{MIMIC_IMAGES_PATH}/{entry["image_path"]}'
            image = MinioUtils.load_image_from_minio(image_path)
        else:
            image_path = os.path.join(str(DATASETS_PATH['mimic-cxr']), MIMIC_IMAGES_PATH, entry['image_path'])
            image = Image.open(image_path).convert('RGB')

        image = transform_image(self.transform, image)

        return {
            'uid': uid,
            'idx': idx,
            'image': image,
            'text': report,
            'labels': labels,
            'caption': caption
        }


class MimicCxrMVSDataset(MimicCxrDataset):
    def __init__(self, args, split, transform, use_minio=False):
        super().__init__(args, split, transform, use_minio)
        nltk.download("punkt_tab")

    def __getitem__(self, idx):
        entry = self.dataset.iloc[idx]
        uid = entry['dicom_id']
        idx = self.uid2idx[uid]

        report = str(entry.get('report', '')).strip()
        labels_df = entry[CHEXPERT_LABELS].copy()
        labels_df[labels_df == -1] = 0
        labels_array = labels_df.to_numpy(dtype=np.float32)
        labels = torch.tensor(labels_array, dtype=torch.float).squeeze(0)

        if self.use_minio:
            image_path = f'mimic-cxr/{MIMIC_IMAGES_PATH}/{entry["image_path"]}'
            image = MinioUtils.load_image_from_minio(image_path)
        else:
            image_path = os.path.join(str(DATASETS_PATH['mimic-cxr']), MIMIC_IMAGES_PATH, entry['image_path'])
            image = Image.open(image_path).convert('RGB')

        image_original = image.copy()
        image = transform_image(self.transform, image)
        augmented_image = transform_image(self.transform, image_original)

        if self.split == 'train':
            sentences = sent_tokenize(report)
            if len(sentences) > 1:
                random.shuffle(sentences)
                augmented_report = ' '.join(sentences)
            else:
                augmented_report = report
        else:
            augmented_report = report

        return {
            'uid': uid,
            'idx': idx,
            'image': image,
            'image_view': augmented_image,
            'text': report,
            'text2': augmented_report,
            'labels': labels,
        }
