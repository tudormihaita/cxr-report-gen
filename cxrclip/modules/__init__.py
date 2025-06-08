import os
import torch
from typing import Dict
from transformers import BertConfig
from blip.bert import BertLMHeadModel

from .text_encoder import BertTextEncoder
from .prompt_constructor import PromptConstructor, PromptStrategy
from .image_encoder import VisionTransformerEncoder
from .image_classifier import LinearClassifier, MLPClassifier
from .projection import LinearProjectionHead, MLPProjectionHead

from utils.logger import LoggerManager
log = LoggerManager.get_logger(__name__)


def load_image_encoder(config_image_encoder: Dict):
    if config_image_encoder["source"].lower() == "huggingface":
        cache_dir = config_image_encoder[
            "cache_dir"] if "cache_dir" in config_image_encoder else "~/.cache/huggingface/hub"
        gradient_checkpointing = (
            config_image_encoder[
                "gradient_checkpointing"] if "gradient_checkpointing" in config_image_encoder else False
        )
        model_type = config_image_encoder["model_type"] if "model_type" in config_image_encoder else "vit"
        _image_encoder = VisionTransformerEncoder(
            name=config_image_encoder["name"],
            pretrained=config_image_encoder["pretrained"],
            gradient_checkpointing=gradient_checkpointing,
            cache_dir=cache_dir,
            model_type=model_type,
            local_files_only=os.path.exists(
                os.path.join(cache_dir, f'models--{config_image_encoder["name"].replace("/", "--")}')),
        )
    else:
        raise KeyError(f"Not supported image encoder: {config_image_encoder}")
    return _image_encoder


def load_text_encoder(config_text_encoder: Dict, vocab_size: int):
    if config_text_encoder["source"].lower() == "huggingface":
        cache_dir = config_text_encoder["cache_dir"]
        gradient_checkpointing = config_text_encoder["gradient_checkpointing"]
        _text_encoder = BertTextEncoder(
            name=config_text_encoder["name"],
            vocab_size=vocab_size,
            pretrained=config_text_encoder["pretrained"],
            gradient_checkpointing=gradient_checkpointing,
            cache_dir=cache_dir,
            local_files_only=os.path.exists(
                os.path.join(cache_dir, f'models--{config_text_encoder["name"].replace("/", "--")}')),
            trust_remote_code=config_text_encoder["trust_remote_code"],
        )
    else:
        raise KeyError(f"Not supported text encoder: {config_text_encoder}")
    return _text_encoder


def load_projection_head(embedding_dim: int, config_projection_head: Dict):
    model_type = config_projection_head["name"].lower()
    if model_type == "mlp":
        projection_head = MLPProjectionHead(
            embedding_dim=embedding_dim, projection_dim=config_projection_head["proj_dim"],
            dropout=config_projection_head["dropout"]
        )
    elif model_type == "linear":
        projection_head = LinearProjectionHead(embedding_dim=embedding_dim,
                                               projection_dim=config_projection_head["proj_dim"])
    else:
        raise KeyError(f"Not supported text encoder: {config_projection_head}")
    return projection_head


def load_image_classifier(config_image_classifier: Dict, feature_dim: int):
    model_type = config_image_classifier["name"].lower()
    if model_type == "linear":
        _image_classifier = LinearClassifier(feature_dim=feature_dim, num_class=config_image_classifier["n_class"])
    elif model_type == "mlp":
        _image_classifier = MLPClassifier(feature_dim=feature_dim, hidden_dim=config_image_classifier["hidden_dim"],
                                          num_class=config_image_classifier["n_class"])
    else:
        raise KeyError(f"Not supported image classifier: {config_image_classifier}")

    return _image_classifier


def load_text_decoder(config_text_decoder: Dict, vocab_size: int, encoder_hidden_size: int):
    if config_text_decoder["source"].lower() == "huggingface":
        cache_dir = config_text_decoder.get("cache_dir", "~/.cache/huggingface/hub")
        decoder_name = config_text_decoder["name"].lower()

        decoder_config = BertConfig.from_pretrained(
            decoder_name,
            cache_dir=cache_dir,
            vocab_size=vocab_size,
        )

        decoder_config.add_cross_attention = True
        decoder_config.is_decoder = True
        decoder_config.encoder_width = encoder_hidden_size

        text_decoder = BertLMHeadModel(config=decoder_config)
        text_decoder.resize_token_embeddings(vocab_size)

        return text_decoder


def load_prompt_constructor(
        model_config: Dict,
        pretrained_model,
):
    prompt_constructor = PromptConstructor(
        prompt_strategy=model_config["prompt_strategy"],
        pretrained_model=pretrained_model,
        prompt_file_path=model_config["prompt_file_path"],
        optimal_thresholds=model_config["optimal_thresholds"],
        use_diverse_templates=model_config["use_diverse_templates"],
    )
    return prompt_constructor


def load_pretrained_weights(model, model_config: Dict, ckpt_key: str = "model_state_dict"):
    log.info("Loading pre-trained weights")
    if not os.path.isfile(model_config["load_backbone_weights"]):
        raise ValueError(f"Cannot find a weight file: {model_config['load_backbone_weights']}")

    ckpt = torch.load(model_config['load_backbone_weights'], map_location="cpu", weights_only=False)
    if ckpt_key not in ckpt:
        raise KeyError(f"Checkpoint does not contain key '{ckpt_key}'")

    model.load_state_dict(ckpt[ckpt_key], strict=True)
    log.info("Loaded model weights successfully")

    if model_config.get("freeze_backbone_weights", False):
        log.info("Freezing model weights")
        for param in model.parameters():
            param.requires_grad = False

    return model


def load_pretrained_image_encoder_weights(model_config: Dict, ckpt_key: str = "model_state_dict"):
    log.info("Loading pre-trained image encoder weights")
    if not os.path.isfile(model_config["load_backbone_weights"]):
        raise ValueError(f"Cannot find a weight file: {model_config['load_backbone_weights']}")
    ckpt = torch.load(model_config["load_backbone_weights"], map_location="cpu", weights_only=False)
    image_encoder = load_image_encoder(model_config["image_encoder"])
    image_encoder_weights = {}
    for k in ckpt[ckpt_key].keys():
        if k.startswith("image_encoder."):
            image_encoder_weights[".".join(k.split(".")[1:])] = ckpt[ckpt_key][k]
    image_encoder.load_state_dict(image_encoder_weights, strict=True)

    if model_config["freeze_backbone_weights"]:
        log.info("Freezing image encoder to not be re-trained")
        for param in image_encoder.parameters():
            param.requires_grad = False

    return image_encoder


def load_pretrained_text_encoder_weights(model_config: Dict, vocab_size: int, ckpt_key: str = "model_state_dict"):
    log.info("Loading pre-trained text encoder weights")
    if not os.path.isfile(model_config["load_backbone_weights"]):
        raise ValueError(f"Cannot find a weight file: {model_config['load_backbone_weights']}")

    ckpt = torch.load(model_config["load_backbone_weights"], map_location="cpu", weights_only=False)
    text_encoder = load_text_encoder(model_config["text_encoder"], vocab_size)

    text_encoder_weights = {}
    for k in ckpt[ckpt_key].keys():
        if k.startswith("text_encoder."):
            text_encoder_weights[".".join(k.split(".")[1:])] = ckpt[ckpt_key][k]

    if text_encoder_weights:
        text_encoder.load_state_dict(text_encoder_weights, strict=True)
        log.info("Loaded text encoder weights successfully")
    else:
        raise ValueError("No text encoder weights found in checkpoint")

    if model_config.get("freeze_backbone_weights", False):
        log.info("Freezing text encoder weights")
        for param in text_encoder.parameters():
            param.requires_grad = False

    return text_encoder


def load_pretrained_classifier_weights(model_config: Dict, feature_dim: int, ckpt_key: str = "model_state_dict"):
    log.info("Loading pre-trained classifier weights")
    if not os.path.isfile(model_config["load_backbone_weights"]):
        raise ValueError(f"Cannot find classifier weight file: {model_config['load_backbone_weights']}")

    ckpt = torch.load(model_config["load_backbone_weights"], map_location="cpu", weights_only=False)
    classifier = load_image_classifier(model_config["classifier"], feature_dim)
    classifier_weights = {}
    for k in ckpt[ckpt_key].keys():
        if k.startswith("classifier."):
            classifier_weights[".".join(k.split(".")[1:])] = ckpt[ckpt_key][k]

    if classifier_weights:
        classifier.load_state_dict(classifier_weights, strict=True)
        log.info("Loaded classifier weights successfully")
    else:
        raise ValueError("No classifier weights found in checkpoint")

    if model_config.get("freeze_backbone_weights", False):
        log.info("Freezing classifier weights")
        for param in classifier.parameters():
            param.requires_grad = False

    return classifier
