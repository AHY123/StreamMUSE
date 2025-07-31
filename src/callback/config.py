from ..utils.base_config import Field, dataclass, ConfigDict
from pydantic import model_validator
from typing import Optional, Literal, Union

@dataclass(config=ConfigDict(extra="allow"))
class ModelCheckpointCallbackConfig:
    """
    Schema for the ModelCheckpoint callback configuration.
    """
    _target_: Literal["pytorch_lightning.callbacks.ModelCheckpoint"] = "pytorch_lightning.callbacks.ModelCheckpoint"
    dirpath: str = Field("./checkpoints", description="Directory where model checkpoints will be saved.")
    filename: Optional[str] = Field("{epoch:02d}-{step:05d}-{val_loss:.2f}", description="Name of the checkpoint file. Optional if not needed.")
    monitor: str = Field("val_loss", description="Metric to monitor for saving checkpoints.")
    mode: Literal["min", "max"] = Field("min", description="Mode for monitoring metric, either 'min' or 'max'.")
    save_top_k: int = Field(5, description="Number of top k checkpoints to save based on monitored metric.")
    save_last: bool = Field(True, description="Whether to save the last checkpoint.")
    
    @model_validator(mode="after")
    def post_init(self):
        """
        Validate the configuration after initialization.
        """
        if not self.dirpath:
            raise ValueError("dirpath must be specified for ModelCheckpointCallbackConfig.")
        if not isinstance(self.save_top_k, int) or self.save_top_k < 0:
            raise ValueError("save_top_k must be a non-negative integer.")
        return self
    

UnionCallbackConfig = Union[ModelCheckpointCallbackConfig]