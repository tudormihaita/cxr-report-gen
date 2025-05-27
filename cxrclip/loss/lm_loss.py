import torch.nn as nn


class LanguageModelingLoss(nn.Module):
    def __init__(self, ignore_index=-100, loss_ratio=1.0):
        super(LanguageModelingLoss, self).__init__()
        self.name = "language_modeling"
        self.ignore_index = ignore_index
        self.loss_ratio = loss_ratio
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index,
            reduction="mean"
        )

    def forward(self, logits, labels, model_loss=None, **kwargs):
        if model_loss is not None:
            return model_loss * self.loss_ratio
        else:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss * self.loss_ratio