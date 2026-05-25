import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.projection = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.projection(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings