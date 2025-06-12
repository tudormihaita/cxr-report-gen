"""
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
"""
import warnings
warnings.filterwarnings("ignore")

import os
import torch
from urllib.parse import urlparse
from timm.models.hub import download_cached_file

from .model import BLIP
from .vit import interpolate_pos_embed
from .downstream import BLIPFeatureExtractor, BLIPDecoder, BLIPRetrieval


def blip_decoder(pretrained=None, use_custom=False, **kwargs):
    model = BLIPDecoder(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained, use_custom)
        print(f"Missing keys: {msg.missing_keys}")
    return model

def blip_feature_extractor(pretrained=None, use_custom=False, **kwargs):
    model = BLIPFeatureExtractor(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained, use_custom)
        print(f"Missing keys: {msg.missing_keys}")
    return model

def blip_pretrain(pretrained=None, use_custom=False, **kwargs):
    model = BLIP(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained, use_custom)
        print(f"Missing keys: {msg.missing_keys}")
    return model

def blip_retrieval(pretrained=None, use_custom=False, **kwargs):
    model = BLIPRetrieval(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained, use_custom)
        print(f"Missing keys: {msg.missing_keys}")
    return model


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

def load_checkpoint(model, url_or_filename, use_custom):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu')
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('Checkpoint URL or path is invalid')

    if use_custom:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint['model']

    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],
                                                                   model.visual_encoder)
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                         model.visual_encoder_m)
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape != model.state_dict()[key].shape:
                del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    print('Loaded checkpoint from %s' % url_or_filename)
    return model, msg
