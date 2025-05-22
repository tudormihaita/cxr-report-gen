import torch
import torch.nn as nn
from torch.nn import functional as F


def soft_clip_loss(image_embeddings, text_embeddings, soft_targets, logit_scale):
    """Compute soft contrastive loss using KL divergence."""
    logits_i2t = logit_scale * image_embeddings @ text_embeddings.T
    logits_t2i = logit_scale * text_embeddings @ image_embeddings.T

    log_probs_i2t = F.log_softmax(logits_i2t, dim=1)
    log_probs_t2i = F.log_softmax(logits_t2i, dim=1)

    loss_i2t = F.kl_div(log_probs_i2t, soft_targets, reduction='batchmean')
    loss_t2i = F.kl_div(log_probs_t2i, soft_targets, reduction='batchmean')

    return loss_i2t, loss_t2i


def label_similarity(labels):
    """Compute similarity between samples based on their labels using Jaccard index."""
    binary_labels = torch.clamp(labels, min=0).float()

    intersection = torch.matmul(binary_labels, binary_labels.T)
    sum_labels = binary_labels.sum(dim=1, keepdim=True)
    union = sum_labels + sum_labels.T - intersection

    jaccard_sim = intersection / (union + 1e-8)

    return jaccard_sim


class ImageTextContrastiveLoss(nn.Module):
    def __init__(self, label_smoothing=0.0, i2i_weight=1.0, t2t_weight=0.5, loss_ratio=1.0):
        super(ImageTextContrastiveLoss, self).__init__()
        self.name = "contrastive"
        self.label_smoothing = label_smoothing
        self.loss_ratio = loss_ratio
        self.i2i_weight = i2i_weight
        self.t2t_weight = t2t_weight

    def forward(self,
                image_embeddings, text_embeddings,
                text_embeddings2, image_view_embeddings,
                logit_scale, concept_labels,
                is_train, **kwargs
                ):
        batch_size = image_embeddings.shape[0]
        device = image_embeddings.device

        label_sim = label_similarity(concept_labels)
        soft_targets = F.softmax(label_sim * logit_scale, dim=1)
        hard_targets = torch.arange(batch_size, device=device)

        loss_i2t, loss_t2i = 0.0, 0.0

        # I1 - T1
        loss_i1t1, loss_t1i1 = soft_clip_loss(image_embeddings, text_embeddings, soft_targets, logit_scale)
        loss_i2t += loss_i1t1
        loss_t2i += loss_t1i1

        # I2 - T1
        loss_i2t1, loss_t1i2 = soft_clip_loss(image_view_embeddings, text_embeddings, soft_targets, logit_scale)
        loss_i2t += loss_i2t1
        loss_t2i += loss_t1i2

        # I1 - T2
        loss_i1t2, loss_t2i1 = soft_clip_loss(image_embeddings, text_embeddings2, soft_targets, logit_scale)
        loss_i2t += loss_i1t2
        loss_t2i += loss_t2i1

        # I2 - T2
        loss_i2t2, loss_t2i2 = soft_clip_loss(image_view_embeddings, text_embeddings2, soft_targets, logit_scale)
        loss_i2t += loss_i2t2
        loss_t2i += loss_t2i2

        loss_i2t = loss_i2t / 4.0
        loss_t2i = loss_t2i / 4.0

        # ICL (Image Contrastive Learning)
        loss_i2i = 0

        logits_per_i2i1 = logit_scale * image_embeddings @ image_view_embeddings.T
        logits_per_i1i2 = logit_scale * image_view_embeddings @ image_embeddings.T

        loss_i2i += F.cross_entropy(logits_per_i2i1, hard_targets)
        loss_i2i += F.cross_entropy(logits_per_i1i2, hard_targets)
        loss_i2i = loss_i2i / 2.0

        # TCL (Text Contrastive Learning)
        loss_t2t = 0

        logits_per_t2t1 = logit_scale * text_embeddings2 @ text_embeddings.T
        logits_per_t1t2 = logit_scale * text_embeddings @ text_embeddings2.T

        loss_t2t += F.cross_entropy(logits_per_t2t1, hard_targets)
        loss_t2t += F.cross_entropy(logits_per_t1t2, hard_targets)
        loss_t2t = loss_t2t / 2.0

        # Combined contrastive loss
        loss = (loss_i2t + loss_t2i) / 2.0
        loss += loss_i2i * self.i2i_weight
        loss += loss_t2t * self.t2t_weight

        return loss.mean()
