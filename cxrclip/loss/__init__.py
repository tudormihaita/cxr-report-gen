from typing import Dict

from .combined_loss import CombinedLoss
from .classification_loss import ClassificationLoss
from .contrastive_loss import SemanticDistributionAlignmentLoss, ImageTextContrastiveLoss, SemanticContrastiveAlignmentLoss
from .lm_loss import LanguageModelingLoss


def build_loss(all_loss_config: Dict) -> CombinedLoss:
    loss_list = []

    for loss_config in all_loss_config:
        cfg = all_loss_config[loss_config]
        if cfg["loss_ratio"] == 0.0:
            continue
        if loss_config == "contrastive_clip":
            loss = ImageTextContrastiveLoss(**cfg)
        elif loss_config == "contrastive_distribution":
            loss = SemanticDistributionAlignmentLoss(**cfg)
        elif loss_config == "contrastive_semantic":
            loss = SemanticContrastiveAlignmentLoss(**cfg)
        elif loss_config == "classification":
            loss = ClassificationLoss(**cfg)
        elif loss_config == "language_modeling":
            loss = LanguageModelingLoss(**cfg)
        else:
            raise KeyError(f"Unknown loss: {loss_config}")

        loss_list.append(loss)

    total_loss = CombinedLoss(loss_list)
    return total_loss
