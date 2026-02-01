from torch.utils.data import DataLoader, Dataset
from typing import Optional, Callable
import torch

class DataLoaderFactory:
    """
    Factory for creating high-performance PyTorch DataLoaders.
    Defaults to optimal settings for pin_memory and num_workers.
    """
    @staticmethod
    def create(
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = None
    ) -> DataLoader:
        """
        Creates a DataLoader with performance presets.
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )
