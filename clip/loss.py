import torch
import torch.nn as nn
import torch.nn.functional as F


class CCALoss(nn.Module):
    """
    Contrastive and Conceptual Alignment loss function
    The goal is to align the image-text representations and the medical concepts, ensuring the model learns
    both the textual and conceptual relationships effectively.
    """

    def __init__(
            self,
            temperature: float = 0.07,
            concept_weight: float = 0.5,
            concept_sim_weight: float = 0.3,
            pos_weight=None,
            eval=False,
    ):
        super().__init__()
        self.temperature = temperature
        self.concept_weight = concept_weight
        self.concept_sim_weight = concept_sim_weight
        self.pos_weight = pos_weight

        if not eval:
            self.concept_loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
        else:
            self.concept_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def clip_loss(self, logits_per_image, logits_pert_text, labels=None, label_sim=None):
        if label_sim is not None:
            targets = F.softmax(label_sim / self.temperature, dim=1)
            log_probs_img = F.log_softmax(logits_per_image, dim=1)
            log_probs_text = F.log_softmax(logits_pert_text, dim=1)
            image_loss = F.kl_div(log_probs_img, targets, reduction='batchmean')
            text_loss = F.kl_div(log_probs_text, targets.T, reduction='batchmean')
        else:
            image_loss = F.cross_entropy(logits_per_image, labels)
            text_loss = F.cross_entropy(logits_pert_text, labels)
        return (image_loss + text_loss) / 2.0

    def concept_classification_loss(self, concepts_logits, medical_concepts):
        # masking uncertain labels to ensure loss calculation only on certain labels
        mask = (medical_concepts != -1).float()

        # replacing uncertain labels since BCEWithLogitsLoss supports only binary classification
        targets = medical_concepts.clone()
        targets[targets == -1] = 0.0

        loss = self.concept_loss_fn(concepts_logits, targets.float())
        masked_loss = (loss * mask).sum() / (mask.sum() + 1e-8)

        return masked_loss

    def concept_similarity_loss(self, concepts_image_sim, medical_concepts):
        batch_size = medical_concepts.shape[0]

        concepts_weights  = medical_concepts.clone().float()
        concepts_weights[medical_concepts == -1] = 0.0

        sim_matrix = torch.zeros((batch_size, batch_size), device=concepts_weights.device)

        for i in range(batch_size):
            for j in range(batch_size):
                # computing the Jaccard similarity between pairs of concepts in the batch
                intersection = torch.min(concepts_weights[i], concepts_weights[j]).sum()
                union = torch.max(concepts_weights[i], concepts_weights[j]).sum()

                sim_matrix[i, j] = intersection / union if union > 0 else 0.0

        sim_matrix = sim_matrix / self.temperature
        log_probs = F.log_softmax(concepts_image_sim, dim=1)
        targets = F.softmax(sim_matrix, dim=1)

        loss = F.kl_div(log_probs, targets, reduction='batchmean')
        return loss

    def forward(self, outputs, medical_concepts):
        batch_size = outputs['logits_per_image'].shape[0]
        similarity_matrix_labels = torch.arange(batch_size, device=outputs['logits_per_image'].device)

        clip_loss = self.clip_loss(
            outputs['logits_per_image'],
            outputs['logits_per_text'],
            similarity_matrix_labels
        )

        loss_dict = {
            "clip_loss": clip_loss.item(),
        }
        total_loss = clip_loss

        if self.concept_weight > 0:
            concept_loss = self.concept_classification_loss(
                outputs['concepts_logits'],
                medical_concepts
            )
            total_loss += concept_loss * self.concept_weight
            loss_dict["concept_loss"] = concept_loss.item()

        if self.concept_sim_weight > 0:
            if "concepts_image_similarity" in outputs:
                concept_sim_loss = self.concept_similarity_loss(
                    outputs['concepts_image_similarity'],
                    medical_concepts
                )
                total_loss += self.concept_sim_weight * concept_sim_loss
                loss_dict["concepts_sim_loss"] = concept_sim_loss.item()

        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict


def calculate_concept_similarity(concepts1, concepts2):
    batch_size = concepts1.shape[0]

    concepts1 = concepts1.clone().float()
    concepts1[concepts1 == - 1] = 0.0

    concepts2 = concepts2.clone().float()
    concepts2[concepts2 == - 1] = 0.0

    sim_matrix = torch.zeros((batch_size, batch_size), device=concepts1.device)

    for i in range(batch_size):
        for j in range(batch_size):
            intersection = torch.min(concepts1[i], concepts2[j]).sum()
            union = torch.max(concepts1[i], concepts2[j]).sum()

            sim_matrix[i, j] = intersection / union if union > 0 else 0.0

    return sim_matrix