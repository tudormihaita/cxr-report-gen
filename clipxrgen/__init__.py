from typing import Dict

from torch import nn
from transformers.tokenization_utils import PreTrainedTokenizer

from .clip import CLIPXRad
from .decoder import CLIPXRGen
from .classifier import CLIPXRadClassifier
from .prompt_constructor import PromptStrategy, ConceptPromptBuilder


def build_model(model_config: Dict, tokenizer: PreTrainedTokenizer = None) -> nn.Module:
    if model_config["name"].lower() == "pretrain_encoder":
        model = CLIPXRad(model_config, tokenizer)
    elif model_config["name"].lower() == "finetune_classifier":
        model = CLIPXRadClassifier(model_config)
    elif model_config["name"].lower() == "downstream_decoder":
        model = CLIPXRGen(model_config, tokenizer)
    elif model_config["name"].lower() == "prompt_constructor":
        model = _build_model_from_pretrained_backbone(model_config, tokenizer)
    else:
        raise KeyError(f"Not supported model: {model_config['name']}")
    return model


def _build_model_from_pretrained_backbone(model_config: Dict, tokenizer=None) -> nn.Module:
    prompt_strategy = PromptStrategy(model_config["prompt_strategy"])
    backbone_config = model_config["pretrained_backbone"]
    if backbone_config is None:
        raise ValueError("Missing `pretrained_backbone` in model config.")

    if prompt_strategy in [PromptStrategy.TEACHER_FORCING, PromptStrategy.ZERO_SHOT]:
        backbone = CLIPXRad(backbone_config, tokenizer)
    elif prompt_strategy == PromptStrategy.SUPERVISED:
        backbone = CLIPXRadClassifier(backbone_config)
    else:
        raise ValueError(f"Unsupported prompt strategy: {prompt_strategy}")

    model = ConceptPromptBuilder(
        pretrained_backbone=backbone,
        prompt_strategy=prompt_strategy,
        prompt_file_path=model_config["prompt_file_path"],
        optimal_thresholds=model_config["optimal_thresholds"],
        use_diverse_templates=model_config["use_diverse_templates"],
    )
    return model
