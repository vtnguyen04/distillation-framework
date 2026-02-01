import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from src.core.interfaces import ModelInterface

class PyTorchModelWrapper(nn.Module, ModelInterface):
    """
    Wrapper ensuring any PyTorch nn.Module adheres to the ModelInterface protocol.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Any:
        return self.model(x)

    def get_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Retrieves features. Assumption: Model has a 'forward_features' method
        or returns a dict in forward.
        This is a placeholder for specific model implementations.
        """
        if hasattr(self.model, 'forward_features'):
            return self.model.forward_features(x)
        raise NotImplementedError("Underlying model does not support feature extraction.")

    def to(self, device: torch.device) -> 'PyTorchModelWrapper':
        super().to(device)
        return self

    def eval(self) -> 'PyTorchModelWrapper':
        super().eval()
        return self

    def train(self, mode: bool = True) -> 'PyTorchModelWrapper':
        super().train(mode)
        return self
