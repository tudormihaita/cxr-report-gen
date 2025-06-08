import os
import json
import torch
import numpy as np
import albumentations
import albumentations.pytorch.transforms

from PIL import Image
from torchvision import transforms
from collections import defaultdict
from typing import Union, Dict
from transformers import AutoTokenizer
from constants import CHEXPERT_LABELS


def load_prompts_from_json(prompt_file_path: str, class_name: str):
    with open(prompt_file_path, 'r') as f:
        all_prompts = json.load(f)

    if class_name not in all_prompts:
        return [], []

    pos_prompts = all_prompts[class_name].get("pos", [])
    neg_prompts = all_prompts[class_name].get("neg", [])
    return pos_prompts, neg_prompts


def load_tokenizer(pretrained_model_name_or_path, cache_dir, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        cache_dir=cache_dir,
        local_files_only=os.path.exists(
            os.path.join(cache_dir, f'models--{pretrained_model_name_or_path.replace("/", "--")}')),
        **kwargs,
    )
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.cls_token_id

    if tokenizer.eos_token_id is None:
        if tokenizer.sep_token_id is not None:
            tokenizer.eos_token_id = tokenizer.sep_token_id
        else:
            tokenizer.add_special_tokens({'eos_token': '[EOS]'})

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    return tokenizer


def load_transform(split: str = "train", transform_config: Dict = None):
    assert split in {"train", "val", "test"}

    config = []
    if transform_config:
        if split in transform_config:
            config = transform_config[split]
    image_transforms = []

    for name in config:
        if hasattr(transforms, name):
            tr_ = getattr(transforms, name)
        else:
            tr_ = getattr(albumentations, name)
        tr = tr_(**config[name])
        image_transforms.append(tr)

    return image_transforms


def transform_image(image_transforms, image: Union[Image.Image, np.ndarray]):
    if not image_transforms:
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)(image)
        return image

    for tr in image_transforms:
        if isinstance(tr, albumentations.BasicTransform):
            image = np.array(image) if not isinstance(image, np.ndarray) else image

            if len(image.shape) == 3 and image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))

            image = tr(image=image)["image"]
        else:
            image = transforms.ToPILImage()(image) if not isinstance(image, Image.Image) else image
            image = tr(image)

    if not isinstance(image, torch.Tensor):
        image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)(image)
    return image


def generate_image_caption(labels):
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


def build_image_text_similarity_mappings(texts, uids):
    """
    Create a mapping between semantically identical texts and their corresponding imaging studies.
    This helps in computing retrieval metrics.
    :param texts: list of texts, one for each sample
    :param uids: list of unique identifiers for the images
    :return: dicts mapping image index to list of matching text indices and vice versa
    """
    text_to_images = defaultdict(list)
    for uid, text in zip(uids, texts):
        text = text.strip().lower()
        text_to_images[text].append(uid)

    img2txt = dict()
    for uid, text in zip(uids, texts):
        text = text.strip().lower()
        matching_uids = text_to_images[text]
        img2txt[uid] = matching_uids

    return img2txt
