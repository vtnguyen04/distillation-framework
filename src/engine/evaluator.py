import torch
from typing import Dict, Any, Optional
from accelerate import Accelerator
from tqdm.auto import tqdm
from src.core.interfaces import ModelInterface

class Evaluator:
    """
    Evaluator service for validating student model performance.
    """
    def __init__(
        self,
        model: ModelInterface,
        val_loader: torch.utils.data.DataLoader,
        config: Dict[str, Any] = None
    ):
        self.accelerator = Accelerator()
        self.model, self.val_loader = self.accelerator.prepare(model, val_loader)
        self.config = config or {}

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Runs evaluation on the validation set.
        Returns a dictionary of metrics (e.g., accuracy, loss).
        """
        self.model.eval()
        correct = 0
        total = 0

        pbar = tqdm(self.val_loader, desc="Evaluating", disable=not self.accelerator.is_local_main_process)

        for inputs, targets in pbar:
            outputs = self.model(inputs)

            # Assuming classification for this base implementation
            # This logic should ideally be injected via a Metric Strategy
            predictions = outputs.argmax(dim=1)

            # Gather for distributed evaluation
            predictions, targets = self.accelerator.gather_for_metrics((predictions, targets))

            correct += (predictions == targets).sum().item()
            total += targets.size(0)

        accuracy = correct / total if total > 0 else 0.0

        if self.accelerator.is_local_main_process:
            print(f"Validation Accuracy: {accuracy:.4f}")

        return {"accuracy": accuracy}
