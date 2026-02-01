import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset

from src.core.interfaces import ModelInterface
from src.infra.model_wrapper import PyTorchModelWrapper
from src.infra.loader import DataLoaderFactory
from src.infra.loss import KLDivergenceLoss
from src.engine.trainer import Trainer
from src.engine.evaluator import Evaluator

# 1. Define Simple Models
class SimpleTeacher(nn.Module):
    def forward(self, x):
        return x * 2

class SimpleStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10, bias=False)

    def forward(self, x):
        return self.layer(x)

# 2. Setup
def main():
    # Synthetic Data
    x = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,)) # Dummy labels
    dataset = TensorDataset(x, y)

    # Loaders
    loader = DataLoaderFactory.create(dataset, batch_size=10, num_workers=0) # 0 workers for simple script

    # Models
    teacher = PyTorchModelWrapper(SimpleTeacher())
    student = PyTorchModelWrapper(SimpleStudent())

    # Optimizer & Loss
    optimizer = optim.SGD(student.parameters(), lr=0.01)
    loss_fn = KLDivergenceLoss(temperature=2.0, alpha=0.5)

    # 3. Train
    print("Starting Training...")
    trainer = Trainer(
        student=student,
        teacher=teacher,
        train_loader=loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device="cpu"
    )

    trainer.fit(epochs=2)
    print("Training Complete!")

    # 4. Evaluate
    # Use same loader for eval for simplicity
    evaluator = Evaluator(
        model=student,
        val_loader=loader
    )
    evaluator.evaluate()

if __name__ == "__main__":
    main()
