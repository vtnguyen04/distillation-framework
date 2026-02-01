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

class CSVLogger(Callback):
    """
    Logs metrics to a CSV file for plotting.
    """
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.csv_path = f"{save_dir}/results.csv"
        self.file = None
        self.keys = []

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        if not trainer.accelerator.is_local_main_process:
            return

        # Initialize file headers on first write to capture all potential keys
        combined_metrics = {"epoch": epoch, **metrics}

        if self.file is None:
            self.keys = list(combined_metrics.keys())
            import csv
            self.file = open(self.csv_path, "w", newline="")
            self.writer = csv.DictWriter(self.file, fieldnames=self.keys)
            self.writer.writeheader()

        # Write
        # align keys just in case
        row = {k: combined_metrics.get(k, 0.0) for k in self.keys}
        self.writer.writerow(row)
        self.file.flush()

    def on_train_end(self, trainer):
        if self.file:
            self.file.close()
            # Auto-plot
            from src.utils.plotting import plot_results
            plot_results(self.csv_path, self.save_dir)

class ModelCheckpoint(Callback):
    """
    Saves model checkpoints.
    Strctly saves:
    - checkpoints/best: The best model so far.
    - checkpoints/last: The latest model.
    """
    def __init__(
        self,
        dirpath: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True, # Ignored, always strict in this refined version
        save_last: bool = True, # Ignored, always strict
        top_k: int = 1 # Ignored
    ):
        self.dirpath = dirpath
        self.monitor = monitor
        self.mode = mode

        self.best_score = np.inf if mode == "min" else -np.inf

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        current_score = metrics.get(self.monitor)

        # Fallback
        if current_score is None:
            if "train_loss" in metrics:
                 current_score = metrics["train_loss"]
            else:
                 return

        # Check Best
        is_best = False
        if self.mode == "min":
            is_best = current_score < self.best_score
        else:
            is_best = current_score > self.best_score

        if is_best:
            self.best_score = current_score
            logger.info(f"⭐️ New best model found! Score: {current_score:.4f}")
            self._save(trainer, "best")

        # Always save Last
        self._save(trainer, "last")

    def _save(self, trainer, name: str):
        if trainer.accelerator.is_local_main_process:
            path = f"{self.dirpath}/{name}"
            # Accelerate's save_state overwrites/merges.
            # Ideally we want to clean it first to ensure no stale files, but save_state handles it mostly okay.
            trainer.accelerator.save_state(path)

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
