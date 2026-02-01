from accelerate import Accelerator
from typing import Dict, Any, Optional
import time

class MLTracker:
    """
    Wrapper for experiment tracking (WandB, Tensorboard) via Accelerate.
    Also handles throughput calculation.
    """
    def __init__(self, accelerator: Accelerator, project_name: str = "flux-distill"):
        self.accelerator = accelerator
        self.project_name = project_name
        self.start_time = None
        self.total_samples = 0

        # Initialize trackers (assumes accelerator is already init with log_with)
        if self.accelerator.is_local_main_process:
            self.accelerator.init_trackers(project_name, config=None)

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to all initialized trackers."""
        if self.accelerator.is_local_main_process:
            self.accelerator.log(metrics, step=step)

    def start_epoch(self):
        self.start_time = time.time()
        self.total_samples = 0

    def update_throughput(self, batch_size: int):
        self.total_samples += batch_size

    def end_epoch(self, epoch: int) -> float:
        """Returns images/sec throughput."""
        if self.start_time is None:
            return 0.0

        elapsed = time.time() - self.start_time
        throughput = self.total_samples / elapsed

        if self.accelerator.is_local_main_process:
            self.accelerator.log({"throughput_img_sec": throughput, "epoch": epoch}, step=epoch)

        return throughput

    def finish(self):
        self.accelerator.end_training()
