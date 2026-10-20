"""Training / validation loop with checkpointing on the best macro F1."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from multimodal_transformer.model.fusion_transformer import MultimodalFusionTransformer
from multimodal_transformer.training.metrics import ClassificationReport, classification_report
from multimodal_transformer.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class TrainerConfig:
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    amp: bool = True
    early_stopping_patience: int = 5
    output_dir: str = "runs/default"


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.inference_mode()
def evaluate(
    model: MultimodalFusionTransformer,
    loader: DataLoader,
    device: torch.device,
    class_names: Sequence[str],
    drop_image: bool = False,
) -> ClassificationReport:
    """Runs the model over ``loader``.

    Args:
        drop_image: Masks every image (LiDAR-only), simulating darkness for NFR-05.
    """
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    for batch in loader:
        batch = move_batch(batch, device)
        valid = torch.zeros_like(batch["image_valid"]) if drop_image else batch["image_valid"]
        logits = model(batch["image"], batch["points"], valid)
        y_true += batch["label"].tolist()
        y_pred += logits.argmax(dim=1).tolist()
    return classification_report(y_true, y_pred, class_names)


class Trainer:
    """Optimises the model and keeps the checkpoint with the best validation macro F1."""

    def __init__(
        self,
        model: MultimodalFusionTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        class_names: Sequence[str],
        config: TrainerConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(device)
        self.class_names = list(class_names)
        self.cfg = config
        self.device = device
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
        steps = max(1, config.epochs * len(train_loader))
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=config.lr, total_steps=steps, pct_start=0.1
        )
        self.use_amp = config.amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.history: List[Dict[str, float]] = []
        self.best_f1 = -1.0

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss, correct, seen = 0.0, 0, 0
        progress = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.cfg.epochs}", leave=False)
        for batch in progress:
            batch = move_batch(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(self.device.type, enabled=self.use_amp):
                logits = self.model(batch["image"], batch["points"], batch["image_valid"])
                loss = self.criterion(logits, batch["label"])
            self.scaler.scale(loss).backward()
            # Gradients are unscaled before clipping so the threshold is in real units.
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item() * batch["label"].size(0)
            correct += (logits.argmax(1) == batch["label"]).sum().item()
            seen += batch["label"].size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / seen:.3f}")
        return {"train_loss": total_loss / max(seen, 1), "train_acc": correct / max(seen, 1)}

    def save_checkpoint(self, path: Path, report: Optional[ClassificationReport] = None) -> None:
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "model_config": vars(self.model.config),
                "class_names": self.class_names,
                "metrics": report.as_dict() if report else None,
            },
            path,
        )

    def fit(self) -> List[Dict[str, float]]:
        LOGGER.info("Training on %s, %d train batches", self.device, len(self.train_loader))
        epochs_without_improvement = 0
        for epoch in range(1, self.cfg.epochs + 1):
            start = time.time()
            log = {"epoch": epoch, **self._train_epoch(epoch)}
            report = evaluate(self.model, self.val_loader, self.device, self.class_names)
            log.update({f"val_{k}": v for k, v in report.as_dict().items()})
            log["duration_sec"] = time.time() - start
            self.history.append(log)
            LOGGER.info(
                "epoch %d | loss %.4f | train acc %.4f | val acc %.4f | val macro F1 %.4f",
                epoch, log["train_loss"], log["train_acc"], report.accuracy, report.macro_f1,
            )  # fmt: skip

            if report.macro_f1 > self.best_f1:
                self.best_f1 = report.macro_f1
                epochs_without_improvement = 0
                self.save_checkpoint(self.output_dir / "best.pth", report)
                LOGGER.info("New best macro F1 %.4f\n%s", report.macro_f1, report.to_text())
            else:
                epochs_without_improvement += 1

            with open(self.output_dir / "training_history.json", "w", encoding="utf-8") as fh:
                json.dump(self.history, fh, indent=2)
            if epochs_without_improvement >= self.cfg.early_stopping_patience:
                LOGGER.info("Early stopping after %d epochs without improvement", epoch)
                break
        self.save_checkpoint(self.output_dir / "last.pth")
        return self.history
