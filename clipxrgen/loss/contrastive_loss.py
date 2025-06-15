import torch
import torch.nn as nn
from torch.nn import functional as F


class ImageTextContrastiveLoss(nn.Module):
    def __init__(self, label_smoothing=0.0, i2i_weight=1.0, t2t_weight=0.5, loss_ratio=1.0):
        super(ImageTextContrastiveLoss, self).__init__()
        self.name = "contrastive_clip"
        self.label_smoothing = label_smoothing
        self.loss_ratio = loss_ratio
        self.i2i_weight = i2i_weight
        self.t2t_weight = t2t_weight

    def forward(self, image_embeddings, text_embeddings, text_aug_embeddings, image_view_embeddings, logit_scale, is_train=True, **kwargs):
        labels = torch.arange(image_embeddings.shape[0])

        loss_i2t = 0
        loss_t2i = 0
        label_smoothing = self.label_smoothing if is_train else 0.0

        # I1 - T1
        logits_per_image, logits_per_text = self._cosine_sim(image_embeddings, text_embeddings, logit_scale)
        loss_i2t += F.cross_entropy(logits_per_image, labels, label_smoothing=label_smoothing)
        loss_t2i += F.cross_entropy(logits_per_text, labels, label_smoothing=label_smoothing)

        # I2 - T1
        logits_per_image, logits_per_text = self._cosine_sim(image_view_embeddings, text_embeddings, logit_scale)
        loss_i2t += F.cross_entropy(logits_per_image, labels, label_smoothing=label_smoothing)
        loss_t2i += F.cross_entropy(logits_per_text, labels, label_smoothing=label_smoothing)

        # I1 - T2
        logits_per_image, logits_per_text = self._cosine_sim(image_embeddings, text_aug_embeddings, logit_scale)
        loss_i2t += F.cross_entropy(logits_per_image, labels, label_smoothing=label_smoothing)
        loss_t2i += F.cross_entropy(logits_per_text, labels, label_smoothing=label_smoothing)

        # I2 - T2
        logits_per_image, logits_per_text = self._cosine_sim(image_view_embeddings, text_aug_embeddings, logit_scale)
        loss_i2t += F.cross_entropy(logits_per_image, labels, label_smoothing=label_smoothing)
        loss_t2i += F.cross_entropy(logits_per_text, labels, label_smoothing=label_smoothing)

        loss_i2t = loss_i2t / 4.0
        loss_t2i = loss_t2i / 4.0

        # ICL
        loss_i2i = 0

        logits_per_i2i1 = logit_scale * image_embeddings @ image_view_embeddings.T
        logits_per_i1i2 = logit_scale * image_view_embeddings @ image_embeddings.T

        loss_i2i += F.cross_entropy(logits_per_i2i1, labels)
        loss_i2i += F.cross_entropy(logits_per_i1i2, labels)
        loss_i2i = loss_i2i / 2.0

        # TCL
        loss_t2t = 0

        logits_per_t2t1 = logit_scale * text_aug_embeddings @ text_embeddings.T
        logits_per_t1t2 = logit_scale * text_embeddings @ text_aug_embeddings.T

        loss_t2t += F.cross_entropy(logits_per_t2t1, labels)
        loss_t2t += F.cross_entropy(logits_per_t1t2, labels)
        loss_t2t = loss_t2t / 2.0

        loss = (loss_i2t + loss_t2i) / 2.0  # shape: (batch_size,)
        loss += loss_i2i * self.i2i_weight
        loss += loss_t2t * self.t2t_weight
        loss = loss * self.loss_ratio

        return loss.mean()

    def _cosine_sim(self, image_embeddings, text_embeddings, logit_scale):
        logits_per_image = logit_scale * image_embeddings @ text_embeddings.T
        logits_per_text = logit_scale * text_embeddings @ image_embeddings.T
        return logits_per_image, logits_per_text


class SemanticContrastiveAlignmentLoss(nn.Module):
    def __init__(self, semantic_temperature=1.0, semantic_weight=0.7, loss_ratio=1.0):
        super(SemanticContrastiveAlignmentLoss, self).__init__()
        self.name = "contrastive_mcsl"
        self.semantic_temperature = semantic_temperature
        self.semantic_weight = semantic_weight
        self.loss_ratio = loss_ratio

    def forward(self,
                image_embeddings, text_embeddings,
                image_view_embeddings, text_aug_embeddings,
                logit_scale, concept_labels,
                is_train=True, **kwargs
                ):
        device = image_embeddings.device
        label_sim = self._jaccard_sim(concept_labels).to(device)
        label_sim.fill_diagonal_(0.0)  # prevent self-matching

        total_loss = 0.0
        # I1 - T1
        total_loss += self._soft_clip_loss(image_embeddings, text_embeddings, label_sim, logit_scale)
        # I2 - T1
        total_loss += self._soft_clip_loss(image_view_embeddings, text_embeddings, label_sim, logit_scale)
        # I1 - T2
        total_loss += self._soft_clip_loss(text_embeddings, text_aug_embeddings, label_sim, logit_scale)
        # I2 - T2
        total_loss += self._soft_clip_loss(image_view_embeddings, text_aug_embeddings, label_sim, logit_scale)

        total_loss = total_loss / 4.0
        total_loss = total_loss * self.loss_ratio
        return total_loss.mean()

    def _soft_clip_loss(self, img_embed, text_embed, label_sim, logit_scale):
        batch_size = img_embed.shape[0]
        device = img_embed.device
        logits = logit_scale * torch.matmul(img_embed, text_embed.T)

        exact_targets = torch.eye(batch_size, device=device)
        semantic_targets = F.softmax(label_sim * self.semantic_temperature, dim=1)
        soft_targets = exact_targets + (semantic_targets * self.semantic_weight)
        soft_targets = F.normalize(soft_targets, p=1, dim=-1)

        loss_i2t = self._soft_xent_loss(logits, soft_targets)
        loss_t2i = self._soft_xent_loss(logits.T, soft_targets.T)
        return (loss_i2t + loss_t2i) / 2.0

    def _soft_xent_loss(self, inputs, targets):
        logprobs = F.log_softmax(inputs, dim=1)
        return -(targets * logprobs).sum() / inputs.shape[0]

    def _jaccard_sim(self, labels):
        binary_labels = torch.clamp(labels, min=0).float()

        intersection = torch.matmul(binary_labels, binary_labels.T)
        sum_labels = binary_labels.sum(dim=1, keepdim=True)
        union = sum_labels + sum_labels.T - intersection

        jaccard_sim = intersection / (union + 1e-8)

        return jaccard_sim


class SemanticDistributionAlignmentLoss(nn.Module):
    def __init__(self, loss_ratio=1.0):
        super(SemanticDistributionAlignmentLoss, self).__init__()
        self.name = "contrastive_kldiv"
        self.loss_ratio = loss_ratio

    def forward(self,
                image_embeddings, text_embeddings,
                text_aug_embeddings, image_view_embeddings,
                logit_scale, concept_labels,
                is_train=True, **kwargs
                ):
        label_sim = self._jaccard_sim(concept_labels)
        soft_targets = F.softmax(label_sim * logit_scale, dim=1)

        loss_i2t, loss_t2i = 0.0, 0.0

        # I1 - T1
        loss_i1t1, loss_t1i1 = self._kl_div_loss(image_embeddings, text_embeddings, soft_targets, logit_scale)
        loss_i2t += loss_i1t1
        loss_t2i += loss_t1i1

        # I2 - T1
        loss_i2t1, loss_t1i2 = self._kl_div_loss(image_view_embeddings, text_embeddings, soft_targets, logit_scale)
        loss_i2t += loss_i2t1
        loss_t2i += loss_t1i2

        # I1 - T2
        loss_i1t2, loss_t2i1 = self._kl_div_loss(image_embeddings, text_aug_embeddings, soft_targets, logit_scale)
        loss_i2t += loss_i1t2
        loss_t2i += loss_t2i1

        # I2 - T2
        loss_i2t2, loss_t2i2 = self._kl_div_loss(image_view_embeddings, text_aug_embeddings, soft_targets, logit_scale)
        loss_i2t += loss_i2t2
        loss_t2i += loss_t2i2

        loss_i2t = loss_i2t / 4.0
        loss_t2i = loss_t2i / 4.0

        loss = (loss_i2t + loss_t2i) / 2.0
        loss = loss * self.loss_ratio
        return loss.mean()

    def _kl_div_loss(self, image_embeddings, text_embeddings, soft_targets, logit_scale):
        logits_i2t = logit_scale * image_embeddings @ text_embeddings.T
        logits_t2i = logit_scale * text_embeddings @ image_embeddings.T

        log_probs_i2t = F.log_softmax(logits_i2t, dim=1)
        log_probs_t2i = F.log_softmax(logits_t2i, dim=1)

        loss_i2t = F.kl_div(log_probs_i2t, soft_targets, reduction='batchmean')
        loss_t2i = F.kl_div(log_probs_t2i, soft_targets, reduction='batchmean')

        return loss_i2t, loss_t2i

    def _jaccard_sim(self, labels):
        binary_labels = torch.clamp(labels, min=0).float()

        intersection = torch.matmul(binary_labels, binary_labels.T)
        sum_labels = binary_labels.sum(dim=1, keepdim=True)
        union = sum_labels + sum_labels.T - intersection

        jaccard_sim = intersection / (union + 1e-8)

        return jaccard_sim