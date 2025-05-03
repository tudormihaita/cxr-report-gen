import os
import torch
import pandas as pd
import numpy as np

from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms import ToTensor

from utils.minio import MinioUtils
from constants import DATASETS_PATH, CHEXPERT_LABELS, IU_XRAY_IMAGES_PATH, MIMIC_IMAGES_PATH


class IUXrayDataset(Dataset):
    def __init__(self, args, split_path, transform=None):
        self.args = args
        self.dataset = pd.read_csv(os.path.join(str(DATASETS_PATH['iu-xray']), 'processed', split_path))
        self.transform = transform if transform else ToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        entry = self.dataset.iloc[idx]
        uid = entry['uid']

        report = str(entry['impression'])
        labels_array = (entry[CHEXPERT_LABELS].replace(-1, 0)).to_numpy(dtype=np.float32)
        labels = torch.tensor(labels_array, dtype=torch.float).squeeze(0)
        caption = _generate_image_caption(labels_array)
        # TODO: load both frontal and lateral projections if present
        proj_path = entry['frontal_filename'] if pd.notna(entry['frontal_filename']) else entry['lateral_filename']
        image = Image.open(os.path.join(str(DATASETS_PATH['iu-xray']), IU_XRAY_IMAGES_PATH, proj_path)).convert('RGB')
        image = self.transform(image)

        return uid, report, image, labels, caption


class MimicCXRDataset(Dataset):
    def __init__(self, args, split, transform=None, use_minio=False):
        self.args = args
        self.use_minio = use_minio
        self.transform = transform if transform else ToTensor()

        if self.use_minio:
            self.dataset = MinioUtils.load_csv_from_minio(f'mimic-cxr/processed/mimic-cxr_{split}.csv')
        else:
            self.dataset = pd.read_csv(
                os.path.join(str(DATASETS_PATH['mimic-cxr']), 'processed-cxr-pro', f'mimic-cxr_{split}.csv'))

        reports = (self.dataset['report'].fillna('')).astype(str).tolist()
        captions = [_generate_image_caption(labels) for labels in (self.dataset[CHEXPERT_LABELS].replace(-1, 0)).to_numpy()]
        uids = (self.dataset['dicom_id']).astype(str).tolist()

        self.img2txt, self.txt2img = _build_text_image_mappings(reports, uids)
        self.img2caption  = _build_caption_image_mappings(captions, uids)
        self.uid2idx = {uid: idx for idx, uid in enumerate(set(self.dataset['dicom_id'].astype(str)))}

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
        caption = _generate_image_caption(labels_array)
        if self.use_minio:
            image_path = f'mimic-cxr/{MIMIC_IMAGES_PATH}/{entry["image_path"]}'
            image = MinioUtils.load_image_from_minio(image_path)
        else:
            image_path = os.path.join(str(DATASETS_PATH['mimic-cxr']), MIMIC_IMAGES_PATH, entry['image_path'])
            image = Image.open(image_path).convert('RGB')

        image = self.transform(image)

        return uid, report, image, labels, caption, idx


def _generate_image_caption(labels):
    """
    Generate a caption for the image based on the labels.
    :param labels: list of CheXpert labels for the image
    :return: generated caption
    """
    findings = []
    for value, label in zip(labels, CHEXPERT_LABELS):
        if value == 1.0:
            findings.append(label)

    if len(findings) == 0:
        findings.append('No Finding')

    findings_caption = ', '.join(findings)
    return "Chest X-ray showing " + findings_caption + "."

def _build_text_image_mappings(reports, uids):
    """
    Create a mapping between semantically identical reports and their corresponding imaging studies.
    This helps in computing retrieval metrics.
    :param reports: list of text reports, one for each sample
    :param uids: list of unique identifiers for the images
    :return: dicts mapping image index to list of matching text indices and vice versa
    """
    report_to_images = defaultdict(list)

    for uid, report in zip(uids, reports):
        report = report.strip().lower()
        report_to_images[report].append(uid)

    img2txt = dict()
    txt2img = dict()

    for uid, report in zip(uids, reports):
        report = report.strip().lower()
        matching_uids = report_to_images[report]

        img2txt[uid] = matching_uids
        txt2img[uid] = uid

    return img2txt, txt2img


def _build_caption_image_mappings(captions, uids):
    """
    Create a mapping between captions and their corresponding images.
    This groups images with identical captions (same medical findings).
    :param captions: list of image captions, one for each sample
    :param uids: list of unique identifiers for the images
    :return: dict mapping image uid to list of image uids with same caption
    """
    caption_to_images = defaultdict(list)

    for uid, caption in zip(uids, captions):
        clean_caption = caption.strip().lower()
        caption_to_images[clean_caption].append(uid)

    img2txt = {}

    for uid, caption in zip(uids, captions):
        clean_caption = caption.strip().lower()
        img2txt[uid] = caption_to_images[clean_caption]

    return img2txt

def _build_weighted_sampler(split, label_columns=CHEXPERT_LABELS):
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
