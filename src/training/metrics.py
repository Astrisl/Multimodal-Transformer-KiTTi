"""Classification metrics: confusion matrix, per-class precision / recall / F1 (NFR-03)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> np.ndarray:
    """Matrix [C, C]; row = true class, column = predicted class."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)), 1)
    return matrix


@dataclass
class ClassificationReport:
    class_names: List[str]
    confusion: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    support: np.ndarray

    @property
    def accuracy(self) -> float:
        total = self.confusion.sum()
        return float(np.trace(self.confusion) / total) if total else 0.0

    @property
    def macro_f1(self) -> float:
        """Unweighted mean over classes present in the ground truth.

        Macro (not micro/weighted) F1 is the target metric because every class must be
        recognised well, not only the dominant Car class.
        """
        present = self.support > 0
        return float(self.f1[present].mean()) if present.any() else 0.0

    def as_dict(self) -> Dict[str, float]:
        out = {"accuracy": self.accuracy, "macro_f1": self.macro_f1}
        for i, name in enumerate(self.class_names):
            out[f"f1_{name}"] = float(self.f1[i])
        return out

    def to_text(self) -> str:
        lines = [f"{'class':<12}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}"]
        for i, name in enumerate(self.class_names):
            lines.append(
                f"{name:<12}{self.precision[i]:>10.4f}{self.recall[i]:>10.4f}"
                f"{self.f1[i]:>10.4f}{self.support[i]:>10d}"
            )
        lines.append(f"accuracy {self.accuracy:.4f} | macro F1 {self.macro_f1:.4f}")
        return "\n".join(lines)


def classification_report(
    y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]
) -> ClassificationReport:
    """Computes the report; undefined ratios (0/0) are reported as 0."""
    matrix = confusion_matrix(y_true, y_pred, len(class_names))
    tp = np.diag(matrix).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    support = matrix.sum(axis=1)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return ClassificationReport(list(class_names), matrix, precision, recall, f1, support)
