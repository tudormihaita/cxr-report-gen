import torch
import torch.nn as nn


class BaseChexpertClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_classes):
        super(BaseChexpertClassifier, self).__init__()
        self.num_classes = num_classes

        self.model = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        logits = self.model(x)
        probs = torch.sigmoid(logits)

        outputs = {
            'logits': logits,
            'probs': probs
        }

        return outputs


class ChexpertXRClassifier(BaseChexpertClassifier):
    def __init__(self, vision_model, embed_dim, hidden_dim, num_classes):
        super(ChexpertXRClassifier, self).__init__(embed_dim, hidden_dim, num_classes)
        self.vision_model = vision_model

    def forward(self, x):
        _, x_embed = self.vision_model(x)
        return super().forward(x_embed)


class ChexpertConceptClassifier(BaseChexpertClassifier):
    def __init__(self, embed_dim, hidden_dim, num_classes):
        super(ChexpertConceptClassifier, self).__init__(embed_dim, hidden_dim, num_classes)