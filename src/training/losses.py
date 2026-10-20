"""Weighted cross-entropy against class imbalance.

KITTI is dominated by cars (~28k Car/Van/Truck vs ~4.5k Pedestrian vs ~1.6k Cyclist
objects). Unweighted cross-entropy reaches high *accuracy* by favouring Car while the
rare classes stay poorly recognised, which is what macro F1 exposes.

The weighted loss multiplies each sample's loss by the weight of its true class::

    L = - sum_i w[y_i] * log p(y_i | x_i) / sum_i w[y_i]

so one Cyclist error costs as much as ``w_cyclist / w_car`` Car errors.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn


def inverse_frequency_weights(class_counts: Sequence[int], power: float = 0.5) -> torch.Tensor:
    """Weights ``w_c ∝ (N / n_c) ** power``, rescaled so the most frequent class has w = 1.

    ``power = 1`` is full inverse frequency; it over-weights very rare classes and makes
    training noisy, so the square root (``power = 0.5``) is the default compromise.
    Empty classes get weight 0 (nothing to learn from them).
    """
    counts = torch.tensor(class_counts, dtype=torch.float32)
    weights = torch.zeros_like(counts)
    present = counts > 0
    weights[present] = (counts.sum() / counts[present]) ** power
    return weights / weights[present].min()


def build_loss(
    class_weights: Union[str, Sequence[float], None],
    class_counts: Optional[Sequence[int]] = None,
    label_smoothing: float = 0.0,
) -> nn.CrossEntropyLoss:
    """Creates the training criterion.

    Args:
        class_weights: Explicit list (e.g. ``[1, 5, 10]`` as in v1), ``"auto"`` to derive
            them from ``class_counts``, or None for an unweighted loss.
        class_counts: Training-set class histogram, required for ``"auto"``.
        label_smoothing: Passed to :class:`torch.nn.CrossEntropyLoss`.
    """
    if class_weights is None:
        weight = None
    elif class_weights == "auto":
        if class_counts is None:
            raise ValueError("class_counts is required for class_weights='auto'")
        weight = inverse_frequency_weights(class_counts)
    else:
        weight = torch.tensor(list(class_weights), dtype=torch.float32)
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
