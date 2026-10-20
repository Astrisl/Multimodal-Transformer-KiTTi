"""Training entry point.

Usage::

    python -m multimodal_transformer.training.train \\
        --dataset dataset.yaml --model model.yaml --train train.yaml
"""

from __future__ import annotations

import argparse
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from multimodal_transformer.data_pipeline.dataset import KittiMultimodalDataset
from multimodal_transformer.model.fusion_transformer import build_model
from multimodal_transformer.training.losses import build_loss
from multimodal_transformer.training.trainer import Trainer, TrainerConfig
from multimodal_transformer.utils.config import load_yaml
from multimodal_transformer.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_datasets(data_cfg: Dict, num_points: int, augment: bool = True):
    common = dict(
        crops_dir=data_cfg["crops_dir"],
        class_names=data_cfg["class_names"],
        val_ratio=data_cfg.get("val_ratio", 0.2),
        seed=data_cfg.get("split_seed", 42),
        num_points=num_points,
        intensity_scale=data_cfg.get("intensity_scale", 1.0),
        image_size=data_cfg.get("image_size", 224),
    )
    train = KittiMultimodalDataset(split="train", augment=augment, **common)
    val = KittiMultimodalDataset(split="val", augment=False, **common)
    return train, val


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train the multimodal fusion Transformer")
    parser.add_argument("--dataset", default="dataset.yaml")
    parser.add_argument("--model", default="model.yaml")
    parser.add_argument("--train", default="train.yaml")
    args = parser.parse_args(argv)

    data_cfg = load_yaml(args.dataset)
    model_cfg = load_yaml(args.model)["model"]
    train_cfg = load_yaml(args.train)
    seed_everything(train_cfg.get("seed", 42))

    model_cfg["num_classes"] = len(data_cfg["class_names"])
    train_ds, val_ds = build_datasets(data_cfg, num_points=train_cfg.get("num_points", 1024))
    LOGGER.info(
        "Train %d / val %d objects, train class counts %s",
        len(train_ds), len(val_ds), dict(zip(data_cfg["class_names"], train_ds.class_counts())),
    )  # fmt: skip

    loader_args = dict(
        batch_size=train_cfg.get("batch_size", 16),
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    criterion = build_loss(
        train_cfg.get("class_weights"),
        class_counts=train_ds.class_counts(),
        label_smoothing=train_cfg.get("label_smoothing", 0.0),
    )
    LOGGER.info("Class weights: %s", criterion.weight)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer_cfg = TrainerConfig(
        **{k: v for k, v in train_cfg.items() if k in TrainerConfig.__dataclass_fields__}
    )
    trainer = Trainer(
        build_model(model_cfg), train_loader, val_loader, criterion,
        data_cfg["class_names"], trainer_cfg, device,
    )  # fmt: skip
    trainer.fit()


if __name__ == "__main__":
    main()
