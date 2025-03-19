import torch
import torch.nn as nn

class ChexpertClassifier(nn.Module):
    def __init__(self, vision_model, embed_dim, hidden_dim, num_classes, threshold=0.5):
        super(ChexpertClassifier, self).__init__()
        self.num_classes = num_classes
        self.threshold = threshold

        self.vision_model = vision_model
        self.model = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x_proj, x_embed = self.vision_model(x)
        logits = self.model(x_embed)
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).int()

        outputs = {
            'logits': logits,
            'probs': probs,
            'preds': preds
        }

        return outputs
