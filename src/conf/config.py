from dataclasses import dataclass, field
from typing import Optional, List, Any
from hydra.core.config_store import ConfigStore

@dataclass
class DataConfig:
    batch_size: int = 128
    num_workers: int = 4
    dataset_name: str = "cifar10"
    data_root: str = "./data"
    pin_memory: bool = True

@dataclass
class ModelConfig:
    teacher_architecture: str = "resnet18"
    student_architecture: str = "resnet18_half"
    pretrained_teacher: bool = True

@dataclass
class LossConfig:
    type: str = "kl" # kl, attention, contrastive
    temperature: float = 4.0
    alpha: float = 0.5 # Balance between distillation and student loss
    beta: float = 1000.0 # Weight for attention transfer
    contrastive_weight: float = 0.8 # Weight for CRD

@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 5e-4
    device: str = "cuda"
    save_dir: str = "./checkpoints"
    log_interval: int = 10

@dataclass
class DistillationConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

# Register configs
cs = ConfigStore.instance()
cs.store(name="base_config", node=DistillationConfig)
