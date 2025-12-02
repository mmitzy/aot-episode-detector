# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class AOTEpisodeNet(nn.Module):
    """
    ResNet18 backbone + 2 heads:
      1) Classifier head: episode classification
      2) Embedding head: feature vector for anomaly detection
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 128,
        backbone: str = "resnet18",
        pretrained: bool = True,
    ):
        super().__init__()

        # ===== Backbone (ResNet18) =====
        if backbone == "resnet18":
            try:
                # Newer torchvision API
                from torchvision.models import ResNet18_Weights
                weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                base = models.resnet18(weights=weights)
            except Exception:
                # Older API
                base = models.resnet18(pretrained=pretrained)

            in_dim = base.fc.in_features
            base.fc = nn.Identity()  # remove the 1000-class head
            self.backbone = base

        else:
            raise NotImplementedError(f"Backbone '{backbone}' not implemented.")

        # ===== Head 1: Classifier =====
        self.classifier_head = nn.Linear(in_dim, num_classes)

        # ===== Head 2: Embedding =====
        self.embed_head = nn.Linear(in_dim, embed_dim)

    def forward(self, x, return_features: bool = False):
        # Feature vector from ResNet
        feats = self.backbone(x)              # shape: [B, in_dim]

        # Classification head
        logits = self.classifier_head(feats)  # [B, num_classes]

        # Embedding head
        embedding = self.embed_head(feats)    # [B, embed_dim]
        embedding = F.normalize(embedding, dim=-1)

        if return_features:
            return logits, embedding, feats

        return logits, embedding
