from ..utils.base_config import BaseConfig, Field, dataclass, ConfigDict
from typing import Any, Optional

@dataclass(config=ConfigDict(extra="allow"))
class LRSchedulerConfig:
    """Config for learning rate scheduler configuration."""

    _target_: str = Field("torch.optim.lr_scheduler.CosineAnnealingLR", description="Learning rate scheduler class")
    optimizer = Field(..., description="Optimizer object")
    T_max: int = Field(1e4, description="Total number of training steps")
    eta_min: float = Field(1e-6, description="Minimum learning rate")
