"""Token builder: (image patch, object points) -> model inputs.

Used unchanged by the training dataset, the evaluation scripts and the real-time ROS 2
inference node, so the preprocessing seen at training time is exactly the one used live.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from multimodal_transformer.data_pipeline.normalization import PointCloudNormalizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class EvalImageTransform:
    """Resize + ToTensor + Normalize, the same operations as the torchvision pipeline,
    with a batched path: patches are resized one by one (PIL), then converted and
    normalised as one tensor instead of one small tensor per object."""

    def __init__(self, image_size: int = 224) -> None:
        self.resize = transforms.Resize((image_size, image_size))
        self.mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    def resized(self, image: Image.Image) -> np.ndarray:
        return np.asarray(self.resize(image))

    def batch(self, resized: Sequence[np.ndarray]) -> torch.Tensor:
        uint8 = torch.from_numpy(np.stack(resized)).permute(0, 3, 1, 2).contiguous()
        return uint8.float().div(255).sub(self.mean).div(self.std)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.batch([self.resized(image)])[0]


def build_image_transform(image_size: int = 224, train: bool = False) -> Callable:
    """Image preprocessing for the ImageNet-pretrained visual encoder.

    Training adds photometric jitter only: geometric flips would have to be mirrored in
    the point cloud to keep the two modalities consistent.
    """
    if not train:
        return EvalImageTransform(image_size)
    steps = [transforms.Resize((image_size, image_size))]
    if train:
        steps.append(transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3))
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


def to_rgb(image: np.ndarray, encoding: str = "bgr8") -> np.ndarray:
    """Converts BGR / RGB / single-channel IR images to 3-channel RGB uint8.

    IR (``mono8`` / ``mono16``) frames are replicated to three channels so the same
    pretrained RGB encoder can be used for both camera types.
    """
    if image.ndim == 2 or encoding.startswith("mono"):
        mono = image if image.ndim == 2 else image[..., 0]
        if mono.dtype != np.uint8:
            mono = cv2.normalize(mono, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return np.repeat(mono[..., None], 3, axis=2)
    if encoding == "rgb8":
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@dataclass
class MultimodalSampleBuilder:
    """Builds one model sample.

    Attributes:
        normalizer: Point cloud normaliser ([N, 4] -> [num_points, 4]).
        image_transform: PIL -> tensor [3, S, S].
        image_size: S, used for the placeholder tensor of a missing image.
        min_image_brightness: Mean grey level (0..255) below which an image patch is
            treated as missing - e.g. an RGB camera at night. The model then relies on
            LiDAR alone (NFR-05) instead of being fed an uninformative black patch.
    """

    normalizer: PointCloudNormalizer
    image_transform: Callable
    image_size: int = 224
    min_image_brightness: float = 0.0

    def image_is_usable(self, image_rgb: Optional[np.ndarray]) -> bool:
        if image_rgb is None or image_rgb.size == 0:
            return False
        return float(image_rgb.mean()) >= self.min_image_brightness

    def build(self, image_rgb: Optional[np.ndarray], points: np.ndarray) -> Dict[str, torch.Tensor]:
        """Returns ``{"image": [3,S,S], "points": [N,4], "image_valid": scalar 0/1}``."""
        if self.image_is_usable(image_rgb):
            image = self.image_transform(Image.fromarray(image_rgb))
            valid = 1.0
        else:
            image = torch.zeros(3, self.image_size, self.image_size)
            valid = 0.0
        return {
            "image": image,
            "points": torch.from_numpy(self.normalizer(points)),
            "image_valid": torch.tensor(valid, dtype=torch.float32),
        }

    def build_batch(
        self, images_rgb: Sequence[Optional[np.ndarray]], point_sets: Sequence[np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """Stacks :meth:`build` results for several objects of one frame."""
        if not isinstance(self.image_transform, EvalImageTransform):
            samples = [self.build(img, pts) for img, pts in zip(images_rgb, point_sets)]
            return {key: torch.stack([s[key] for s in samples]) for key in samples[0]}
        size = self.image_size
        valid = [self.image_is_usable(img) for img in images_rgb]
        images = torch.zeros(len(valid), 3, size, size)
        usable = [i for i, ok in enumerate(valid) if ok]
        if usable:
            resized = [self.image_transform.resized(Image.fromarray(images_rgb[i])) for i in usable]
            images[usable] = self.image_transform.batch(resized)
        points = np.stack([self.normalizer(pts) for pts in point_sets])
        return {
            "image": images,
            "points": torch.from_numpy(points),
            "image_valid": torch.tensor(valid, dtype=torch.float32),
        }
