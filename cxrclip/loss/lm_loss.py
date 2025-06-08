import torch.nn as nn


class LanguageModelingLoss(nn.Module):
    def __init__(self, label_smoothing=0.1, ignore_index=-100, loss_ratio=1.0):
        super(LanguageModelingLoss, self).__init__()
        self.name = "language_modeling"
        self.ignore_index = ignore_index
        self.loss_ratio = loss_ratio
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index,
            reduction="mean",
            label_smoothing=label_smoothing,
        )

    def forward(self, logits, labels, output_loss=None, **kwargs):
        if output_loss is not None:
            return output_loss * self.loss_ratio
        else:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss * self.loss_ratio