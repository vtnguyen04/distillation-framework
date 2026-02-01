import torch
from typing import Optional, Dict, Any
from accelerate import Accelerator
from tqdm.auto import tqdm
import logging

from src.core.interfaces import ModelInterface, DistillationLoss
from src.infra.loader import DataLoaderFactory

logger = logging.getLogger(__name__)

class Trainer:
    """
    Distillation Trainer utilizing HuggingFace Accelerate for distributed training.
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
        device: str = "cpu"  # Will be overridden by accelerator
    ):
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        # Prepare components with accelerator
        self.student, self.teacher, self.optimizer, self.train_loader, self.val_loader, self.scheduler = \
            self.accelerator.prepare(
                student, teacher, optimizer, train_loader, val_loader, scheduler
            )

        self.loss_fn = loss_fn
        self.config = config or {}
        self.teacher.eval() # Teacher is always in eval mode

    def train_epoch(self, epoch: int):
        self.student.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, disable=not self.accelerator.is_local_main_process)
        pbar.set_description(f"Epoch {epoch}")

        for batch_idx, (inputs, targets) in enumerate(pbar):
            with self.accelerator.accumulate(self.student):
                # Forward
                student_out = self.student(inputs)

                with torch.no_grad():
                    teacher_out = self.teacher(inputs)

                # Compute Loss
                # Note: Currently loss_fn doesn't support unpacked intermediate maps in this MVP
                loss = self.loss_fn.compute(student_out, teacher_out, target=targets)

                # Backward
                self.accelerator.backward(loss)

                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

        return total_loss / len(self.train_loader)

    def fit(self, epochs: int):
        logger.info(f"Starting training for {epochs} epochs")
        for epoch in range(1, epochs + 1):
            avg_loss = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}")

            if self.val_loader:
                # Add validation logic here
                pass

            # Checkpoint (simplified)
            if self.accelerator.is_local_main_process:
                self.accelerator.save_state(f"checkpoint_epoch_{epoch}")

    def save_model(self, path: str):
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.student)
        self.accelerator.save(unwrapped_model.state_dict(), path)
