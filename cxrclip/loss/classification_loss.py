import torch.nn as nn


class ClassificationLoss(nn.Module):
    def __init__(self, loss_ratio=1.0, **kwargs):
        super(ClassificationLoss, self).__init__()
        self.name = "classification"
        self.loss_ratio = loss_ratio
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, cls_pred, target_class, **kwargs):
        loss = self.bce(cls_pred, target_class)
        return loss
