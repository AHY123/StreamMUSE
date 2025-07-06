from ..utils.base_config import BaseConfig, Field, dataclass, ConfigDict
from typing import Any, Optional

@dataclass(config=ConfigDict(extra="allow"))
class OptimizerConfig:
    _target_: str = Field("torch.optim.Adam", description="Optimizer class")
    lr: float = Field(1e-5, description="Learning rate")
    betas: tuple[float, float] = Field((0.9, 0.999), description="Betas")
    eps: float = Field(1e-8, description="Epsilon")
    weight_decay: float = Field(0.0, description="Weight decay")
