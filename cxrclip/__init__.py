from typing import Dict

from torch import nn
from transformers.tokenization_utils import PreTrainedTokenizer

from .clip import CxrCLIP
from .classifier import CxrCLIPClassifier
from .decoder import CxrCLIPDecoder


def build_model(model_config: Dict, loss_config: Dict, tokenizer: PreTrainedTokenizer = None, use_custom: bool = False) -> nn.Module:
    if model_config["name"].lower() == "pretrain_encoder":
        model = CxrCLIP(model_config, loss_config, tokenizer)
    elif model_config["name"].lower() == "finetune_classifier":
        model_type = model_config["image_encoder"]["model_type"] if "model_type" in model_config["image_encoder"] else "vit"
        model = CxrCLIPClassifier(model_config, model_type, use_custom=use_custom)
    elif model_config["name"].lower() == "downstream_decoder":
        model = CxrCLIPDecoder(model_config, tokenizer, use_custom=use_custom)
    else:
        raise KeyError(f"Not supported model: {model_config['name']}")
    return model
