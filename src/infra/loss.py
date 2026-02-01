import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from src.core.interfaces import DistillationLoss

class KLDivergenceLoss(DistillationLoss):
    """
    Standard Knowledge Distillation Loss using KL Divergence.
    Based on Hinton et al., "Distilling the Knowledge in a Neural Network".
    """
    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def compute(
        self,
        student_out: torch.Tensor,
        teacher_out: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        intermediate_maps: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:

        # Soft targets
        soft_student = F.log_softmax(student_out / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_out / self.temperature, dim=1)

        distillation_loss = self.kl_div(soft_student, soft_teacher) * (self.temperature ** 2)

        # Hard targets (if available)
        student_loss = 0.0
        if target is not None:
            student_loss = self.ce_loss(student_out, target)

        return self.alpha * distillation_loss + (1.0 - self.alpha) * student_loss

class MSEFeatureLoss(DistillationLoss):
    """
    Feature-based distillation using MSE Loss.
    """
    def __init__(self, beta: float = 1.0):
        self.beta = beta
        self.mse = nn.MSELoss()

    def compute(
        self,
        student_out: Any,
        teacher_out: Any,
        target: Optional[torch.Tensor] = None,
        intermediate_maps: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        if intermediate_maps is None:
            return torch.tensor(0.0, device=student_out.device if isinstance(student_out, torch.Tensor) else None)

        loss = 0.0
        # intermediate_maps should be { 'student_layer_name': student_feat, 'teacher_layer_name': teacher_feat }
        # Ideally, we need a more structured mapping.
        # For now, let's assume specific keys or passed directly.
        # This is a simplification; robust mapping needed for production.

        # Checking if specific keys (s_feat, t_feat) exist for simplicity in this MVP
        if 's_feat' in intermediate_maps and 't_feat' in intermediate_maps:
             loss += self.mse(intermediate_maps['s_feat'], intermediate_maps['t_feat'])

        return self.beta * loss
