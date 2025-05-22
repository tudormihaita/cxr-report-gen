import os
import torch

from torch import nn
from typing import Dict, TypeVar

from utils.logger import LoggerManager
from .modules import load_image_classifier, load_image_encoder

T = TypeVar("T", bound="Module")
log = LoggerManager.get_logger(__name__)


class CxrCLIPClassifier(nn.Module):
    def __init__(self, model_config: Dict, model_type: str = "vit", use_custom: bool = False):
        super().__init__()
        self.model_type = model_type
        self.model_config = model_config
        if model_config["load_backbone_weights"] is None:
            self.image_encoder = load_image_encoder(model_config["image_encoder"])
        else:
            log.info("Loading pre-trained image encoder for fine-tuning")
            if not os.path.isfile(model_config["load_backbone_weights"]):
                raise ValueError(f"Cannot find a weight file: {model_config['load_backbone_weights']}")
            ckpt = torch.load(model_config["load_backbone_weights"], map_location="cpu", weights_only=False)
            model_key = "model_state_dict" if use_custom else "model"
            print(model_config["image_encoder"])
            self.image_encoder = load_image_encoder(model_config["image_encoder"])
            image_encoder_weights = {}
            for k in ckpt[model_key].keys():
                if k.startswith("image_encoder."):
                    image_encoder_weights[".".join(k.split(".")[1:])] = ckpt[model_key][k]
            self.image_encoder.load_state_dict(image_encoder_weights, strict=True)

        if model_config["freeze_backbone_weights"]:
            log.info("Freezing image encoder to not be re-trained")
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        self.classifier = load_image_classifier(model_config["classifier"]["config"], self.image_encoder.out_dim)

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

        # get image features and predict
        image_feature = self.encode_image(batch["images"].to(device))
        cls_pred = self.classifier(image_feature)

        out = {"cls_pred": cls_pred, "target_class": batch["labels"].to(device)}
        return out
