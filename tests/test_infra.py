import torch
import pytest
from src.infra.loss import KLDivergenceLoss, MSEFeatureLoss

def test_kl_divergence_loss():
    loss_fn = KLDivergenceLoss(temperature=1.0, alpha=0.5)

    student_out = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    teacher_out = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([0, 1])

    loss = loss_fn.compute(student_out, teacher_out, target=target)

    assert loss is not None
    assert loss.item() >= 0.0

    loss.backward()
    assert student_out.grad is not None

def test_mse_feature_loss():
    loss_fn = MSEFeatureLoss(beta=1.0)

    s_feat = torch.rand(2, 64, requires_grad=True)
    t_feat = torch.rand(2, 64)

    maps = {'s_feat': s_feat, 't_feat': t_feat}
    loss = loss_fn.compute(None, None, intermediate_maps=maps)

    assert loss is not None
    loss.backward()
    assert s_feat.grad is not None
