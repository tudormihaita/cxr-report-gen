import os
import torch
from torch import nn
from typing import Dict, TypeVar

from utils.logger import LoggerManager
from .modules import load_image_classifier, load_image_encoder, \
    load_pretrained_image_encoder_weights, load_hf_checkpoint

T = TypeVar("T", bound="Module")
log = LoggerManager.get_logger(__name__)

class CLIPXRadClassifier(nn.Module):
    def __init__(self, model_config: Dict):
        super().__init__()
        self.model_config = model_config
        self.image_encoder = load_image_encoder(model_config["image_encoder"])
        self.classifier = load_image_classifier(model_config["classifier"]["config"], self.image_encoder.out_dim)

        if model_config["load_pretrained_weights"]:
            log.info("Loading pretrained weights for fine-tuned classifier setup")

            ckpt_path = model_config["load_pretrained_weights"]
            ckpt = load_hf_checkpoint(ckpt_path, map_location="cpu")
            if "model_state_dict" not in ckpt:
                raise KeyError(f"Checkpoint does not contain key 'model_state_dict'")

            self.load_state_dict(ckpt["model_state_dict"], strict=True)
            if model_config.get("freeze_backbone_weights", False):
                log.info("Freezing model weights")
                for param in self.parameters():
                    param.requires_grad = False
        elif model_config["load_backbone_weights"]:
            self.image_encoder = load_pretrained_image_encoder_weights(model_config, "model_state_dict")

    def encode_image(self, image):
        image_features = self.image_encoder(image)

        global_features = image_features[:, 0]
        return global_features

    def train(self: T, mode: bool = True) -> T:
        if not isinstance(mode, bool):
            raise ValueError("Training mode is expected to be boolean")

        if mode:
            self.image_encoder.eval()
            self.classifier.train()
        else:
            self.image_encoder.eval()
            self.classifier.eval()

        return self

    def forward(self, batch, device=None):
        device = batch["images"].device if device is None else device

        image_features = self.encode_image(batch["images"].to(device))
        cls_pred = self.classifier(image_features)

        out = {"cls_pred": cls_pred, "target_class": batch["labels"].to(device)}
        return out
