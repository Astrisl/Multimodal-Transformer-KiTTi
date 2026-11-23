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


def init_from_checkpoint(model: torch.nn.Module, path: str) -> None:
    """Starts from a trained checkpoint, also when the new model has more classes.

    Rows of the output layer for classes the checkpoint already knows are copied; rows
    of new classes (e.g. ``Background``) keep their random initialisation.
    """
    source = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    target = model.state_dict()
    for name, value in source.items():
        if name not in target:
            continue
        if target[name].shape == value.shape:
            target[name] = value
        elif target[name].shape[1:] == value.shape[1:] and target[name].shape[0] > value.shape[0]:
            target[name][: value.shape[0]] = value
        else:
            raise ValueError(f"{name}: {tuple(value.shape)} -> {tuple(target[name].shape)}")
    model.load_state_dict(target)
    LOGGER.info("Initialised from %s", path)


def build_datasets(data_cfg: Dict, num_points: int, augment: bool = True):
    common = dict(
        crops_dir=data_cfg["crops_dir"],
        class_names=data_cfg["class_names"],
        val_ratio=data_cfg.get("val_ratio", 0.2),
        seed=data_cfg.get("split_seed", 42),
        num_points=num_points,
        intensity_scale=data_cfg.get("intensity_scale", 1.0),
        image_size=data_cfg.get("image_size", 224),
        index_file=data_cfg.get("index_file", "index.csv"),
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
    model = build_model(model_cfg)
    if train_cfg.get("init_checkpoint"):
        init_from_checkpoint(model, train_cfg["init_checkpoint"])
    trainer = Trainer(
        model, train_loader, val_loader, criterion,
        data_cfg["class_names"], trainer_cfg, device,
    )  # fmt: skip
    trainer.fit()


if __name__ == "__main__":
    main()
