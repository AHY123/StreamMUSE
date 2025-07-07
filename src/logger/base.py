from ..utils.base_config import Field, dataclass, ConfigDict
from typing import Optional, Literal, Union


@dataclass(config=ConfigDict(extra="allow"))
class CSVLoggerConfig:
    """
    Schema for the CSV logger configuration.
    """

    _target_: Literal["pytorch_lightning.loggers.CSVLogger"] = "pytorch_lightning.loggers.CSVLogger"
    save_dir: str = Field("./logs", description="Directory where CSV logs will be saved.")
    name: Optional[str] = Field(None, description="Name of the CSV log file. Optional if not needed.")
    version: Optional[str] = Field(None, description="Version of the CSV log file. Optional if not needed.")


@dataclass(config=ConfigDict(extra="allow"))
class WandbLoggerConfig:
    """
    Schema for the Weights & Biases logger configuration.
    """
    _target_ :Literal["pytorch_lightning.loggers.WandbLogger"] = "pytorch_lightning.loggers.WandbLogger"
    project: str = Field(..., description="Name of the Weights & Biases project.")
    entity: Optional[str] = Field(None, description="Entity name for the Weights & Biases project. Optional if not needed.")
    log_model: bool = Field(True, description="Whether to log the model to Weights & Biases. Default is True.")

    save_dir: str = Field("./logs", description="Directory where CSV logs will be saved.")
    name: Optional[str] = Field(None, description="Name of the CSV log file. Optional if not needed.")
    version: Optional[str] = Field(None, description="Version of the CSV log file. Optional if not needed.")


@dataclass(config=ConfigDict(extra="allow"))
class TensorBoardLoggerConfig:
    """
    Schema for the TensorBoard logger configuration.
    """
    _target_ :Literal["pytorch_lightning.loggers.TensorBoardLogger"] = "pytorch_lightning.loggers.TensorBoardLogger"
    save_dir: str = Field("./logs", description="Directory where CSV logs will be saved.")
    name: Optional[str] = Field(None, description="Name of the CSV log file. Optional if not needed.")
    version: Optional[str] = Field(None, description="Version of the CSV log file. Optional if not needed.")


UnionLoggerConfig = Union[TensorBoardLoggerConfig, CSVLoggerConfig, WandbLoggerConfig]
