from torch import nn
from typing import Dict, TypeVar

from .modules import load_image_classifier, load_image_encoder, load_pretrained_image_encoder_weights

T = TypeVar("T", bound="Module")


class CxrCLIPClassifier(nn.Module):
    def __init__(self, model_config: Dict, model_type: str = "vit"):
        super().__init__()
        self.model_type = model_type
        self.model_config = model_config
        if model_config["load_backbone_weights"] is None:
            self.image_encoder = load_image_encoder(model_config["image_encoder"])
        else:
            self.image_encoder = load_pretrained_image_encoder_weights(model_config, "model_state_dict")

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

        image_features = self.encode_image(batch["images"].to(device))
        cls_pred = self.classifier(image_features)

        out = {"cls_pred": cls_pred, "target_class": batch["labels"].to(device)}
        return out
