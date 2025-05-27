import os
from typing import Dict
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    BertConfig, BertLMHeadModel,
    T5Config, T5ForConditionalGeneration,
    BartConfig, BartForConditionalGeneration
)

from .image_classifier import LinearClassifier, MLPClassifier
from .image_encoder import VisionTransformerEncoder
from .projection import LinearProjectionHead, MLPProjectionHead
from .text_encoder import BertTextEncoder


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


def load_text_decoder(config_text_decoder: Dict, vocab_size: int, mode="conditional"):
    if config_text_decoder["source"].lower() == "huggingface":
        cache_dir = config_text_decoder.get("cache_dir", "~/.cache/huggingface/hub")
        pretrained = config_text_decoder.get("pretrained", True)
        decoder_name = config_text_decoder["name"].lower()

        assert mode in ["conditional", "causal"], f"Unsupported text decoder mode: {mode}"
        if mode == "conditional":
            valid_archs = ["t5", "bart", "bert"]
            if not any(arch in decoder_name for arch in valid_archs):
                raise ValueError(f"Conditional generation requires T5, BART, or BERT architecture, got {decoder_name}")
        elif mode == "causal":
            valid_archs = ["gpt", "bert"]
            if not any(arch in decoder_name for arch in valid_archs):
                raise ValueError(f"Causal generation requires GPT or BERT architecture, got {decoder_name}")

        if "t5" in decoder_name:
            decoder_config = T5Config.from_pretrained(
                decoder_name,
                cache_dir=cache_dir,
            )

            if "encoder_width" in config_text_decoder:
                decoder_config.encoder_width = config_text_decoder["encoder_width"]

            if pretrained:
                text_decoder = T5ForConditionalGeneration.from_pretrained(
                    decoder_name,
                    config=decoder_config,
                    cache_dir=cache_dir,
                )
            else:
                text_decoder = T5ForConditionalGeneration.from_config(config=decoder_config)
        elif "bart" in decoder_name:
            decoder_config = BartConfig.from_pretrained(decoder_name, cache_dir=cache_dir)

            if "encoder_width" in config_text_decoder:
                decoder_config.encoder_width = config_text_decoder["encoder_width"]

            if pretrained:
                text_decoder = BartForConditionalGeneration.from_pretrained(
                    config_text_decoder["name"],
                    config=decoder_config,
                    cache_dir=cache_dir,
                )
            else:
                text_decoder = BartForConditionalGeneration(config=decoder_config)
        elif "bert" in decoder_name.lower and mode == "conditional":
            decoder_config = BertConfig.from_pretrained(
                decoder_name,
                cache_dir=cache_dir,
            )
            decoder_config.is_decoder = True
            decoder_config.add_cross_attention = True

            if "encoder_width" in config_text_decoder:
                decoder_config.encoder_width = config_text_decoder["encoder_width"]

            if pretrained:
                text_decoder = BertLMHeadModel.from_pretrained(
                    decoder_name,
                    config=decoder_config,
                    cache_dir=cache_dir,
                )
            else:
                text_decoder = BertLMHeadModel(config=decoder_config)
        elif mode == "causal":
            decoder_config = AutoConfig.from_pretrained(decoder_name, cache_dir=cache_dir)

            if pretrained:
                text_decoder = AutoModelForCausalLM.from_pretrained(
                    config_text_decoder["name"],
                    config=decoder_config,
                    cache_dir=cache_dir,
                )
            else:
                text_decoder = AutoModelForCausalLM.from_config(config=decoder_config)
        else:
            decoder_config = AutoConfig.from_pretrained(
                decoder_name,
                cache_dir=cache_dir,
            )
            if not getattr(decoder_config, "is_encoder_decoder", False):
                raise ValueError(f"Model '{decoder_name}' is not an encoder-decoder model.")

            if pretrained:
                text_decoder = AutoModelForSeq2SeqLM.from_pretrained(
                    decoder_name,
                    config=decoder_config,
                    cache_dir=cache_dir,
                )
            else:
                text_decoder = AutoModelForSeq2SeqLM.from_config(config=decoder_config)
        try:
            text_decoder.resize_token_embeddings(vocab_size)
        except Exception as e:
            print(f"Warning: Could not resize token embeddings: {e}")
    else:
        raise NotImplementedError(f"Decoder source {config_text_decoder['source']} not supported.")

    return text_decoder
