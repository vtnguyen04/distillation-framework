# FluxDistill

A high-performance, modular AI distillation framework built with PyTorch.

## Features
- **High Performance**: Optimized with prefetching DataLoaders and `accelerate` for multi-GPU/TPU training.
- **Advanced Distillation**: Built-in support for:
  - **KL Divergence** (Knowledge Distillation)
  - **Feature Distillation** (MSE)
  - **Attention Transfer** (Zagoruyko & Komodakis)
  - **Contrastive Distillation** (CRD)
- **MLOps Integrated**: Automatic experiment tracking (Throughput, Loss) compatible with Tensorboard and WandB.
- **Easy Configuration**: Structured, type-safe configuration using Hydra.

## Installation

```bash
# Clone the repository
git clone https://github.com/vtnguyen04/distillation-framework.git
cd distillation-framework

# Install dependencies (Recommend using uv or venv)
pip install -e .
```

## Quick Start

### 1. Training with Configs
Run the included CIFAR-10 benchmark to see it in action:

```bash
python examples/benchmark_cifar10.py train.epochs=10
```

### 2. Custom Usage

```python
from src.engine.trainer import Trainer
from src.infra.loss import KLDivergenceLoss, AttentionTransferLoss

# ... define student, teacher, loader ...

# Initialize Trainer with Accelerate support
trainer = Trainer(
    student=student_model,
    teacher=teacher_model,
    train_loader=train_loader,
    optimizer=optimizer,
    loss_fn=KLDivergenceLoss(),
    project_name="my-distillation-project"
)

# Start High-Performance Training
trainer.fit(epochs=20)
```

## Benchmarks
Achieves **~5000+ img/sec** on standard hardware for ResNet18 distillation (CIFAR-10).
