import torch
from typing import Optional, Dict, Any
from accelerate import Accelerator
from tqdm.auto import tqdm
import logging
import time

from src.core.interfaces import ModelInterface, DistillationLoss
from src.infra.mlops import MLTracker

logger = logging.getLogger(__name__)

class Trainer:
    """
    Distillation Trainer utilizing HuggingFace Accelerate for distributed training.
    Updated with MLOps tracking.
    """
    def __init__(
        self,
        student: ModelInterface,
        teacher: ModelInterface,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: DistillationLoss,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        config: Dict[str, Any] = None,
        project_name: str = "flux-distill"
    ):
        import datetime
        import os

        # Setup Experiment Directory
        self.config = config or {}
        # Handle config object or dict
        exp_name = getattr(config, "experiment_name", None) if config else "default_exp"
        if not exp_name and isinstance(config, dict):
             exp_name = config.get("experiment_name", "default_exp")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = f"experiments/{exp_name}/{timestamp}"
        self.checkpoint_dir = f"{self.output_dir}/checkpoints"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize Accelerator with specific project dir
        self.accelerator = Accelerator(log_with="tensorboard", project_dir=self.output_dir)
        self.device = self.accelerator.device

        # Prepare components
        self.student, self.teacher, self.optimizer, self.train_loader, self.val_loader, self.scheduler = \
            self.accelerator.prepare(
                student, teacher, optimizer, train_loader, val_loader, scheduler
            )

        self.loss_fn = loss_fn
        self.teacher.eval()

        # MLOps Tracker
        self.tracker = MLTracker(self.accelerator, project_name=project_name)

    def train_epoch(self, epoch: int):
        self.student.train()
        total_loss = 0.0
        self.tracker.start_epoch()

        pbar = tqdm(self.train_loader, disable=not self.accelerator.is_local_main_process)
        pbar.set_description(f"Epoch {epoch}")

        for batch_idx, (inputs, targets) in enumerate(pbar):
            batch_size = inputs.size(0)

            with self.accelerator.accumulate(self.student):
                # Forward
                student_out = self.student(inputs)

                with torch.no_grad():
                    teacher_out = self.teacher(inputs)

                loss = self.loss_fn.compute(student_out, teacher_out, target=targets)

                # Backward
                self.accelerator.backward(loss)

                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()
                self.tracker.update_throughput(batch_size)

                # Log step metrics
                if batch_idx % 10 == 0:
                     self.tracker.log_metrics({"train_loss": loss.item()}, step=epoch * len(self.train_loader) + batch_idx)

                pbar.set_postfix({'loss': loss.item()})

        throughput = self.tracker.end_epoch(epoch)
        avg_loss = total_loss / len(self.train_loader)

        if self.accelerator.is_local_main_process:
            logger.info(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Throughput: {throughput:.2f} img/s")

        return avg_loss

    def fit(self, epochs: int):
        logger.info(f"🚀 Starting experiment in: {self.output_dir}")
        for epoch in range(1, epochs + 1):
            avg_loss = self.train_epoch(epoch)

            if self.val_loader:
                 # Simplified evaluation hook
                 pass

            if self.accelerator.is_local_main_process:
                # Save to structured dir
                save_path = f"{self.checkpoint_dir}/epoch_{epoch}"
                self.accelerator.save_state(save_path)

        self.tracker.finish()

    def save_model(self, path: str):
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.student)
        self.accelerator.save(unwrapped_model.state_dict(), path)
