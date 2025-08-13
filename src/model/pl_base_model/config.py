from ...utils.base_config import BaseConfig, Field, dataclass
from typing import Optional,Literal,Dict,Any
from ...optimizer.base import OptimizerConfig
from ...lr_scheduler.base import LRSchedulerConfig


@dataclass
class TrainingProbingLoggerConfig:
    """TrainingProbingLoggerConfig. it doesn't need _target_ to decide the class"""

    loss_jump_threshold_X: float = Field(1.5, description="Threshold for loss jump")
    loss_avg_window_Y: int = Field(10, description="Window size for loss average")
    recording_window_N: int = Field(5, description="Window size for recording")

@dataclass
class PlBaseModelConfig(BaseConfig):
    """
    Config for the Base model configuration.
    """

    _target_: Literal["src.model.pl_base_model.PlBaseModel"] = Field("src.model.pl_base_model.PlBaseModel")
    optimizer_config: Optional[OptimizerConfig] = Field(default_factory=lambda: OptimizerConfig(), description="Configuration for the optimizer.")
    
    lr_scheduler_config: Optional[LRSchedulerConfig] = Field(
        default_factory=lambda: LRSchedulerConfig(), description="Configuration for the learning rate scheduler."
    )
    training_probing_training_logger_config: TrainingProbingLoggerConfig = Field(
        default_factory=lambda: TrainingProbingLoggerConfig(), description="Configuration for the training probing logger."
    )

    model_config = {"arbitrary_types_allowed": True}
