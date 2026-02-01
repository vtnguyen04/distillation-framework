# FluxDistill

A high-performance, modular AI distillation framework built with PyTorch, adhering to SOLID and DDD principles.

## Features
- **SOLID/DDD Architecture**: Modular design for easy extension and maintenance.
- **High Performance**: Optimized DataLoaders and `accelerate` integration for distributed training.
- **Flexible**: Protocols for Teachers, Students, and Loss functions.
- **Ready for Production**: Logging, Checkpointing, and Evaluation built-in.

## Installation

```bash
# Clone the repository
git clone <repo_url>
cd distilation

# Install dependencies (Recommend using a virtualenv)
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

### 1. Define your Models
Implement the `ModelInterface` for your Teacher and Student models.

### 2. Create Config and Loaders
Use `DataLoaderFactory` to create high-performance loaders.

### 3. Train
```python
from src.engine.trainer import Trainer
from src.infra.loss import KLDivergenceLoss

trainer = Trainer(
    student=student_model,
    teacher=teacher_model,
    train_loader=train_loader,
    optimizer=optimizer,
    loss_fn=KLDivergenceLoss()
)
trainer.fit(epochs=10)
```

## Testing
```bash
python3 -m unittest discover tests
```
