import unittest
import torch
from src.infra.loss import KLDivergenceLoss, MSEFeatureLoss

class TestInfra(unittest.TestCase):
    def test_kl_divergence_loss(self):
        loss_fn = KLDivergenceLoss(temperature=1.0, alpha=0.5)

        student_out = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        teacher_out = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        target = torch.tensor([0, 1])

        loss = loss_fn.compute(student_out, teacher_out, target=target)

        self.assertIsNotNone(loss)
        self.assertTrue(loss.item() >= 0.0)

        loss.backward()
        self.assertIsNotNone(student_out.grad)

    def test_mse_feature_loss(self):
        loss_fn = MSEFeatureLoss(beta=1.0)

        s_feat = torch.rand(2, 64, requires_grad=True)
        t_feat = torch.rand(2, 64)

        maps = {'s_feat': s_feat, 't_feat': t_feat}
        loss = loss_fn.compute(None, None, intermediate_maps=maps)

        self.assertIsNotNone(loss)
        loss.backward()
        self.assertIsNotNone(s_feat.grad)

if __name__ == '__main__':
    unittest.main()
