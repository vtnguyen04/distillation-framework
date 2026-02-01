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
        if 's_feat' in intermediate_maps and 't_feat' in intermediate_maps:
             loss += self.mse(intermediate_maps['s_feat'], intermediate_maps['t_feat'])

        return self.beta * loss

class AttentionTransferLoss(DistillationLoss):
    """
    Attention Transfer Loss (Zagoruyko & Komodakis, 2017).
    Forces the student's activation maps to match the teacher's attention maps.
    """
    def __init__(self, beta: float = 1000.0):
        self.beta = beta

    def at(self, x):
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

    def compute(
        self,
        student_out: Any,
        teacher_out: Any,
        target: Optional[torch.Tensor] = None,
        intermediate_maps: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        if intermediate_maps is None:
            return torch.tensor(0.0)

        loss = 0.0
        # Expecting lists of feature maps
        if 'student_feats' in intermediate_maps and 'teacher_feats' in intermediate_maps:
             s_feats = intermediate_maps['student_feats']
             t_feats = intermediate_maps['teacher_feats']

             for s, t in zip(s_feats, t_feats):
                 # Resize if necessary
                 if s.shape[2:] != t.shape[2:]:
                     s = F.interpolate(s, t.shape[2:], mode='bilinear', align_corners=False)
                 loss += (self.at(s) - self.at(t)).pow(2).mean()

        return self.beta * loss

class ContrastiveDistillationLoss(DistillationLoss):
    """
    Contrastive Representation Distillation (CRD).
    Simplified InfoNCE version for representation alignment.
    """
    def __init__(self, temperature: float = 0.07, weight: float = 1.0):
        self.temperature = temperature
        self.weight = weight

    def compute(
        self,
        student_out: Any,
        teacher_out: Any,
        target: Optional[torch.Tensor] = None,
        intermediate_maps: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Assumes student_out and teacher_out are feature vectors (e.g. before classifier).
        """
        # Normalize
        s = F.normalize(student_out, dim=1)
        t = F.normalize(teacher_out, dim=1)

        # InfoNCE
        # Positive pairs: (s_i, t_i)
        logits = torch.matmul(s, t.T) / self.temperature
        labels = torch.arange(s.size(0), device=s.device)

        loss = F.cross_entropy(logits, labels)
        return self.weight * loss
