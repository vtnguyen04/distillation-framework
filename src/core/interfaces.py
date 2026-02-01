from typing import Protocol, Any, Dict, Optional, runtime_checkable
import torch

@runtime_checkable
class ModelInterface(Protocol):
    """Protocol defining the interface for any model (Teacher or Student) in the framework."""

    def forward(self, x: torch.Tensor) -> Any:
        ...

    def get_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Optional: Retrieve intermediate features for feature-based distillation."""
        ...

    def to(self, device: torch.device) -> 'ModelInterface':
        ...

    def eval(self) -> 'ModelInterface':
        ...

    def train(self, mode: bool = True) -> 'ModelInterface':
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...

@runtime_checkable
class DistillationLoss(Protocol):
    """Protocol defining the interface for distillation loss functions."""

    def compute(
        self,
        student_out: Any,
        teacher_out: Any,
        target: Optional[torch.Tensor] = None,
        intermediate_maps: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Compute the distillation loss.

        Args:
            student_out: Output from the student model.
            teacher_out: Output from the teacher model.
            target: Ground truth labels (optional).
            intermediate_maps: Dictionary mapping student features to teacher features (optional).

        Returns:
            Computed loss value.
        """
        ...

class Config(Protocol):
    """Protocol for configuration objects."""
    @property
    def training(self) -> Dict[str, Any]: ...

    @property
    def model(self) -> Dict[str, Any]: ...
