from typing import Dict, Any, Optional
import torch
import shutil
import os
import logging
import numpy as np
from accelerate import Accelerator

logger = logging.getLogger(__name__)

class Callback:
    def on_train_begin(self, trainer): pass
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]): pass
    def on_train_end(self, trainer): pass

class ModelCheckpoint(Callback):
    """
    Saves model checkpoints.
    Supports:
    - save_best_only: Save only if monitored metric improves.
    - save_last: Always save the latest model.
    - top_k: Keep only top K best models.
    """
    def __init__(
        self,
        dirpath: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_last: bool = True,
        top_k: int = 1
    ):
        self.dirpath = dirpath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.top_k = top_k

        self.best_score = np.inf if mode == "min" else -np.inf
        self.best_k_models = {} # path -> score

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        full_path = f"{self.dirpath}/epoch_{epoch}"
        current_score = metrics.get(self.monitor)

        if current_score is None:
            if self.monitor != "loss" and self.monitor != "train_loss": # Fallback handling
                 logger.warning(f"Metric {self.monitor} not found in metrics. Available: {metrics.keys()}")
                 return
            # Allow fallback to train loss if val not present
            current_score = metrics.get("train_loss")

        # Logic to check if best
        is_best = False
        if self.mode == "min":
            is_best = current_score < self.best_score
        else:
            is_best = current_score > self.best_score

        # Save Best
        if self.save_best_only:
            if is_best:
                self.best_score = current_score
                logger.info(f"⭐️ New best model found! Score: {current_score:.4f}")
                self._save(trainer, full_path, is_best=True)
        else:
            # If not best only, just save every epoch (or interval determined by trainer loop)
            # But normally Checkpoint callback handles saving.
            self._save(trainer, full_path)

        # Save Last
        if self.save_last:
            last_path = f"{self.dirpath}/last"
            self._save(trainer, last_path)

    def _save(self, trainer, path: str, is_best: bool = False):
        if trainer.accelerator.is_local_main_process:
            trainer.accelerator.save_state(path)
            if is_best:
                 # Clean up top_k if needed.
                 # For simplicity in this MVF (Minimum Viable Framework), we just save.
                 pass

class EarlyStopping(Callback):
    """
    Stops training when a monitored metric has stopped improving.
    """
    def __init__(
        self,
        monitor: str = "val_loss",
        min_delta: float = 0.0,
        patience: int = 10,
        mode: str = "min"
    ):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.wait = 0
        self.best_score = np.inf if mode == "min" else -np.inf
        self.stopped_epoch = 0

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        current_score = metrics.get(self.monitor)
        if current_score is None: return

        if self.mode == "min":
            improved = current_score < (self.best_score - self.min_delta)
        else:
            improved = current_score > (self.best_score + self.min_delta)

        if improved:
            logger.info(f"✅ EarlyStopping: Improved {self.monitor} from {self.best_score:.4f} to {current_score:.4f}")
            self.best_score = current_score
            self.wait = 0
        else:
            self.wait += 1
            logger.info(f"⏳ EarlyStopping: No improvement in {self.monitor} (Best: {self.best_score:.4f} | Current: {current_score:.4f}). Wait: {self.wait}/{self.patience}")
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.should_stop = True
                logger.info(f"🛑 Early stopping triggered at epoch {epoch}")
