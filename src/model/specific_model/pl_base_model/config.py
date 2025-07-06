from ....utils.base_config import BaseConfig, Field, dataclass,ConfigDict
from typing import Any, Optional
from ...network import UnionNetworkConfig


@dataclass
class TrainingProbingLoggerConfig:
    """TrainingProbingLoggerConfig. it doesn't need _target_ to decide the class"""

    loss_jump_threshold_X: float = Field(1.5, description="Threshold for loss jump")
    loss_avg_window_Y: int = Field(10, description="Window size for loss average")
    recording_window_N: int = Field(5, description="Window size for recording")

@dataclass(config=ConfigDict(extra="allow"))
class OptimizerConfig:
    _target_: str = Field("torch.optim.Adam", description="Optimizer class")
    lr: float = Field(1e-5, description="Learning rate")
    betas: tuple[float, float] = Field((0.9, 0.999), description="Betas")
    eps: float = Field(1e-8, description="Epsilon")
    weight_decay: float = Field(0.0, description="Weight decay")

@dataclass(config=ConfigDict(extra="allow"))
class LRSchedulerConfig(BaseConfig):
    """Config for learning rate scheduler configuration."""
    _target_: str = Field("torch.optim.lr_scheduler.CosineAnnealingLR", description="Learning rate scheduler class")
    optimizer = Field(..., description="Optimizer object")
    params: dict[str, Any] = Field(default_factory=lambda: {"T_max": 10000, "eta_min": 1e-6}, description="Parameters for the LR scheduler.")


class PlBaseModelConfig(BaseConfig):
    """
    Config for the Base model configuration.
    """

    _target_: str = Field("src.model.specific_model.base_model.BaseModel")
    #network_config: Optional[UnionNetworkConfig] = Field(None, description="Configuration for the network.")
    optimizer_config: Optional[OptimizerConfig] = Field(default_factory=lambda: OptimizerConfig(), description="Configuration for the optimizer.")
    lr_scheduler_config: Optional[LRSchedulerConfig] = Field(
        default_factory=lambda: LRSchedulerConfig(), description="Configuration for the learning rate scheduler."
    )
    training_probing_training_logger_config: TrainingProbingLoggerConfig = Field(
        default_factory=lambda: TrainingProbingLoggerConfig(), description="Configuration for the training probing logger."
    )

    model_config = {"arbitrary_types_allowed": True}
