import os

VIT_TYPE = "ViT-B/32"

CLIP_CONTEXT_LENGTH = 77
CLIP_EXTENDED_CONTEXT_LENGTH = 248
CLIP_IMAGE_RESOLUTION = 224
CLIP_VISION_LAYERS = 12
CLIP_VISION_WIDTH = 768
CLIP_PATCH_SIZE = 32
CLIP_VOCAB_SIZE = 49408
CLIP_TRANSFORMER_LAYERS = 12
CLIP_TRANSFORMER_WIDTH = 512
CLIP_TRANSFORMER_HEADS = 8
CLIP_EMBED_DIM = 512


BART_TYPE = "facebook/bart-base"
BART_CONTEXT_LENGTH = 256

ROOT_PATH = '/Volumes/T7 Shield/datasets/'

DATASETS_PATH = {
    'iu-xray': os.path.join(ROOT_PATH, 'iu-xray'),
    'mimic-cxr': os.path.join(ROOT_PATH, 'mimic-cxr')
}

IMAGES_PATH = {
    'iu-xray': os.path.join(DATASETS_PATH['iu-xray'], 'images/images_normalized'),
    'mimic-cxr': os.path.join(DATASETS_PATH['mimic-cxr'], 'images') # to be updated
}

CHEXPERT_LABELS = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Enlarged Cardiomediastinum',
    'Fracture',
    'Lung Lesion',
    'Lung Opacity',
    'No Finding',
    'Pleural Effusion',
    'Pleural Other',
    'Pneumonia',
    'Pneumothorax',
    'Support Devices'
]

