import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveConceptualAlignmentLoss(nn.Module):
    """
    Contrastive and Conceptual Alignment loss function
    The goal is to align the image-text representations and the medical concepts, ensuring the model learns
    both the textual and conceptual relationships effectively.
    """

    def __init__(
            self,
            temperature: float = 0.07,
            use_soft_concept_loss: bool = False,
    ):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.use_soft_concept_loss = use_soft_concept_loss

    def forward(self, outputs, medical_concepts=None):
        logits_per_image, logits_per_text = outputs['logits_per_image'], outputs['logits_per_text']

        if not self.use_soft_concept_loss or medical_concepts is None:
            loss = self.info_nce_loss(logits_per_image, logits_per_text)
        else:
            loss = self.soft_concept_loss(logits_per_image, logits_per_text, medical_concepts)

        return loss, {"clip_loss": loss.item()}

    def info_nce_loss(self, logits_per_image, logits_per_text):
        batch_size = logits_per_image.shape[0]
        device = logits_per_image.device

        labels = torch.arange(batch_size, device=device)
        i2t_loss = self.criterion(logits_per_image, labels)
        t2i_loss = self.criterion(logits_per_text, labels)
        return (i2t_loss + t2i_loss) / 2.0

    def soft_concept_loss(self, logits_per_image, logits_per_text, medical_concepts):
        concept_sim = self.calculate_concept_similarity(medical_concepts, medical_concepts)
        soft_targets = F.softmax(concept_sim / self.temperature, dim=1)

        log_probs_i2t = F.log_softmax(logits_per_image, dim=1)
        log_probs_t2i = F.log_softmax(logits_per_text, dim=1)

        loss_i2t = F.kl_div(log_probs_i2t, soft_targets, reduction='batchmean')
        loss_t2i = F.kl_div(log_probs_t2i, soft_targets, reduction='batchmean')
        return (loss_i2t + loss_t2i) / 2.0

    @staticmethod
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
