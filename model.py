import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class EmbeddingNet(nn.Module):
    
    def __init__(self, embedding_dim=128):
        super(EmbeddingNet, self).__init__()
        # Load pretrained ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove the original classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # up to last conv layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(2048, embedding_dim)

    def forward(self, x):
        # x: (batch, 3, H, W)
        x = self.backbone(x)          # (batch, 2048, H', W')
        x = self.avgpool(x)            # (batch, 2048, 1, 1)
        x = x.view(x.size(0), -1)      # (batch, 2048)
        x = self.projection(x)          # (batch, embedding_dim)
        x = F.normalize(x, p=2, dim=1)  # L2 normalize
        return x