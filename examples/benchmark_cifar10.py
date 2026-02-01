import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18
import hydra
from hydra.core.config_store import ConfigStore
from typing import Dict, Any

from src.conf.config import DistillationConfig
from src.infra.model_wrapper import PyTorchModelWrapper
from src.infra.loader import DataLoaderFactory
from src.infra.loss import KLDivergenceLoss, AttentionTransferLoss
from src.engine.trainer import Trainer

# Register config
cs = ConfigStore.instance()
cs.store(name="cifar_config", node=DistillationConfig)

class StudentResNet(nn.Module):
    """Simplified ResNet for student (just to demonstrate architectural diff)."""
    def __init__(self):
        super().__init__()
        # Using a very small model for speed in demonstration
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.fc = nn.Linear(16 * 32 * 32, 10)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)

@hydra.main(version_base=None, config_name="cifar_config")
def main(cfg: DistillationConfig):
    print(f"Running Benchmark with Batch Size: {cfg.data.batch_size}")

    # 1. Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = datasets.CIFAR10(root=cfg.data.data_root, train=True, download=True, transform=transform)
    train_loader = DataLoaderFactory.create(
        trainset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers
    )

    # 2. Models
    # Teacher: Pretrained ResNet18
    teacher = resnet18(pretrained=True)
    teacher.fc = nn.Linear(512, 10) # Adapt for CIFAR10
    teacher_wrapper = PyTorchModelWrapper(teacher)

    # Student: Custom Small ResNet
    student = StudentResNet()
    student_wrapper = PyTorchModelWrapper(student)

    # 3. Loss & Opt
    # Hybrid Loss: KL + Attention Transfer
    kl_loss = KLDivergenceLoss(temperature=cfg.loss.temperature, alpha=cfg.loss.alpha)
    # Note: Attention Transfer requires feature maps, which our simple wrapper/model doesn't fully expose generically yet.
    # For this benchmark, we'll stick to KL to verify high-performance loop.

    optimizer = optim.SGD(student.parameters(), lr=cfg.train.lr, momentum=cfg.train.momentum)

    # 4. Trainer with MLOps
    trainer = Trainer(
        student=student_wrapper,
        teacher=teacher_wrapper,
        train_loader=train_loader,
        optimizer=optimizer,
        loss_fn=kl_loss,
        config=cfg,
        project_name="cifar10-benchmark"
    )

    trainer.fit(epochs=cfg.train.epochs)

    # 5. Evaluation
    print("\nStarting Evaluation...")
    testset = datasets.CIFAR10(root=cfg.data.data_root, train=False, download=True, transform=transform)
    test_loader = DataLoaderFactory.create(
        testset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers
    )

    from src.engine.evaluator import Evaluator
    evaluator = Evaluator(
        model=student_wrapper,
        val_loader=test_loader,
        config=cfg
    )

    metrics = evaluator.evaluate()
    print(f"Final Test Metrics: {metrics}")

if __name__ == "__main__":
    main()
