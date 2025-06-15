from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self, feature_dim, num_class):
        super().__init__()
        self.classification_head = nn.Linear(feature_dim, num_class)

    def forward(self, x):
        return self.classification_head(x)


class MLPClassifier(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_class, dropout=0.2):
        super().__init__()
        self.classification_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_class),
        )

    def forward(self, x):
        return self.classification_head(x)