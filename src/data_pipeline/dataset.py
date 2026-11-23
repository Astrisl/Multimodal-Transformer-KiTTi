"""PyTorch dataset over the crops produced by :mod:`build_dataset`."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from multimodal_transformer.data_pipeline.normalization import PointCloudNormalizer
from multimodal_transformer.data_pipeline.sample_builder import (
    MultimodalSampleBuilder,
    build_image_transform,
)


def split_by_scene(sample_ids: Sequence[str], val_ratio: float, seed: int) -> Dict[str, List[str]]:
    """Splits *frame* ids into train / val.

    Splitting by frame (not by object) keeps all objects of one scene on the same side,
    otherwise the same car seen in one frame would leak into validation.
    """
    unique = sorted(set(sample_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = int(round(len(unique) * val_ratio))
    return {"val": sorted(unique[:n_val]), "train": sorted(unique[n_val:])}


class PointCloudAugmenter:
    """Geometric augmentation of a normalised cloud [N, 4] (training only).

    Random rotation around the vertical axis makes the model invariant to the object's
    heading, which matters because live OBBs from clustering have an ambiguous yaw.
    """

    def __init__(self, max_scale_delta: float = 0.05, jitter_std: float = 0.01) -> None:
        self.max_scale_delta = max_scale_delta
        self.jitter_std = jitter_std

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        angle = float(np.random.uniform(-np.pi, np.pi))
        c, s = np.cos(angle), np.sin(angle)
        rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=points.dtype)
        scale = 1.0 + float(np.random.uniform(-self.max_scale_delta, self.max_scale_delta))
        out = points.clone()
        out[:, :3] = (points[:, :3] @ rot.T) * scale
        out[:, :3] += torch.randn_like(out[:, :3]) * self.jitter_std
        return out


class KittiMultimodalDataset(Dataset):
    """(image patch, LiDAR crop, label) samples.

    Args:
        crops_dir: Output directory of the dataset builder (contains ``index.csv``).
        class_names: Ordered project classes; index = label id.
        split: "train", "val" or "all".
        val_ratio: Fraction of frames used for validation.
        seed: Split seed.
        num_points: Points per object.
        intensity_scale: See :class:`PointCloudNormalizer`.
        image_size: Visual encoder input size.
        augment: Enables photometric + geometric augmentation.
    """

    def __init__(
        self,
        crops_dir: str,
        class_names: Sequence[str],
        split: str = "train",
        val_ratio: float = 0.2,
        seed: int = 42,
        num_points: int = 1024,
        intensity_scale: float = 1.0,
        image_size: int = 224,
        augment: bool = False,
        index_file: str = "index.csv",
    ) -> None:
        self.crops_dir = Path(crops_dir)
        self.index_file = index_file
        self.class_names = list(class_names)
        self.class_to_id = {name: i for i, name in enumerate(self.class_names)}

        rows = self._read_index()
        if split != "all":
            frames = set(split_by_scene([r["sample_id"] for r in rows], val_ratio, seed)[split])
            rows = [r for r in rows if r["sample_id"] in frames]
        self.rows = [r for r in rows if r["class_name"] in self.class_to_id]

        self.builder = MultimodalSampleBuilder(
            # Fixed seed for validation so metrics are reproducible between epochs.
            normalizer=PointCloudNormalizer(num_points, intensity_scale, None if augment else seed),
            image_transform=build_image_transform(image_size, train=augment),
            image_size=image_size,
        )
        self.augmenter: Optional[PointCloudAugmenter] = PointCloudAugmenter() if augment else None

    def _read_index(self) -> List[Dict[str, str]]:
        index_path = self.crops_dir / self.index_file
        if not index_path.exists():
            raise FileNotFoundError(
                f"{index_path} not found - run data_pipeline.build_dataset first"
            )
        with open(index_path, "r", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def __len__(self) -> int:
        return len(self.rows)

    def labels(self) -> List[int]:
        return [self.class_to_id[r["class_name"]] for r in self.rows]

    def class_counts(self) -> List[int]:
        counts = Counter(self.labels())
        return [counts.get(i, 0) for i in range(len(self.class_names))]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        points = np.load(self.crops_dir / "lidar" / f"{row['name']}.npy")
        image_rgb = None
        if row["has_image"] == "1":
            bgr = cv2.imread(str(self.crops_dir / "images" / f"{row['name']}.png"))
            if bgr is not None:
                image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        sample = self.builder.build(image_rgb, points)
        if self.augmenter is not None:
            sample["points"] = self.augmenter(sample["points"])
        sample["label"] = torch.tensor(self.class_to_id[row["class_name"]], dtype=torch.long)
        return sample
