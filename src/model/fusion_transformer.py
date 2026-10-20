"""Intermediate-fusion Transformer for object classification (FR-04).

Token sequence fed to the Transformer encoder::

    [CLS] + 49 image tokens (7x7 ResNet-18 feature map) + N point tokens

* Image tokens: frozen ImageNet ResNet-18 trunk -> [B, 512, 7, 7] -> linear projection.
* Point tokens: a shared per-point MLP (PointNet-style) maps (x, y, z, intensity) to the
  embedding. No positional encoding is added: a point cloud is an unordered set and
  the coordinates themselves already carry the geometry.
* A learnable modality embedding is added to every token so self-attention can tell
  the modalities apart, and a CLS token aggregates the fused representation.

Fusion happens *inside* the encoder: every layer lets image and LiDAR tokens attend to
each other, which is what distinguishes intermediate fusion from early (raw data) and
late (separate predictions) fusion.

Robustness to a missing camera (NFR-05): ``image_valid`` masks the image tokens out of
attention via ``src_key_padding_mask``; during training a random fraction of samples has
its image masked (modality dropout) so the network learns a LiDAR-only fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

IMAGE_TOKENS = 49  # 7 x 7 feature map of ResNet-18 at 224 x 224 input


@dataclass
class ModelConfig:
    """Architecture hyper-parameters (see ``config/model.yaml``)."""

    num_classes: int = 3
    embed_dim: int = 128
    nhead: int = 8
    num_layers: int = 3
    ff_dim: int = 256
    dropout: float = 0.1
    point_features: int = 4
    pretrained_backbone: bool = True
    freeze_backbone: bool = True
    modality_dropout: float = 0.2


class ImageEncoder(nn.Module):
    """ResNet-18 trunk -> sequence of image tokens [B, 49, embed_dim]."""

    def __init__(self, embed_dim: int, pretrained: bool, freeze: bool) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # drop avgpool + fc
        self.frozen = freeze
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.projection = nn.Linear(512, embed_dim)

    def train(self, mode: bool = True) -> "ImageEncoder":
        super().train(mode)
        # A frozen backbone must also keep its BatchNorm statistics frozen, otherwise
        # running means drift towards the small KITTI patches even though no weight is
        # being optimised.
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image)  # [B, 512, 7, 7]
        tokens = features.flatten(2).transpose(1, 2)  # [B, 49, 512]
        return self.projection(tokens)


class PointEncoder(nn.Module):
    """Shared per-point MLP: [B, N, F] -> [B, N, embed_dim]."""

    def __init__(self, in_features: int, embed_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.mlp(points)


class MultimodalFusionTransformer(nn.Module):
    """Classifies an object from its image patch and its LiDAR points."""

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = cfg = config or ModelConfig()
        self.image_encoder = ImageEncoder(
            cfg.embed_dim, cfg.pretrained_backbone, cfg.freeze_backbone
        )
        self.point_encoder = PointEncoder(cfg.point_features, cfg.embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.modality_embedding = nn.Parameter(torch.zeros(2, cfg.embed_dim))  # 0 image, 1 lidar
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.modality_embedding, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.nhead,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Nested tensors are disabled: they change output layout and break ONNX export.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Sequential(
            nn.Linear(cfg.embed_dim, 64),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(64, cfg.num_classes),
        )

    def forward(
        self,
        image: torch.Tensor,
        points: torch.Tensor,
        image_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns class logits [B, num_classes].

        Args:
            image: [B, 3, 224, 224] normalised image patches.
            points: [B, N, 4] normalised object point clouds.
            image_valid: [B] (bool or 0/1 float). 0 masks the image tokens out.
        """
        batch = points.shape[0]
        if image_valid is None:
            image_valid = torch.ones(batch, device=points.device)
        image_valid = image_valid.reshape(batch) > 0.5
        if self.training and self.config.modality_dropout > 0:
            keep = torch.rand(batch, device=points.device) >= self.config.modality_dropout
            image_valid = image_valid & keep

        img_tokens = self.image_encoder(image) + self.modality_embedding[0]
        pts_tokens = self.point_encoder(points) + self.modality_embedding[1]
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, img_tokens, pts_tokens], dim=1)

        # True = "ignore this key". CLS and LiDAR tokens are always attended to, so a
        # sample can never end up with every key masked (which would yield NaNs).
        n_img, n_pts = img_tokens.shape[1], pts_tokens.shape[1]
        ignore_img = (~image_valid).unsqueeze(1).expand(batch, n_img)
        keep_rest = torch.zeros(batch, 1 + n_pts, dtype=torch.bool, device=points.device)
        padding_mask = torch.cat([keep_rest[:, :1], ignore_img, keep_rest[:, 1:]], dim=1)

        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        return self.head(self.norm(encoded[:, 0]))


def build_model(config: dict) -> MultimodalFusionTransformer:
    """Creates the model from the ``model`` section of a YAML config."""
    fields = ModelConfig.__dataclass_fields__
    return MultimodalFusionTransformer(
        ModelConfig(**{k: v for k, v in config.items() if k in fields})
    )


def load_checkpoint(path: str, device: str = "cpu") -> MultimodalFusionTransformer:
    """Loads a checkpoint written by the trainer (model config + weights + class names)."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(checkpoint["model_config"], pretrained_backbone=False)
    model = MultimodalFusionTransformer(ModelConfig(**cfg))
    model.load_state_dict(checkpoint["state_dict"])
    model.class_names = checkpoint.get("class_names")
    return model.to(device).eval()
