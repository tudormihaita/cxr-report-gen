import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss

class FocalBCELoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalBCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        target_labels = targets.clone()
        target_labels[target_labels == -1] = 0.0

        return sigmoid_focal_loss(
            inputs=logits,
            targets=target_labels,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction
        )

    def _forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
