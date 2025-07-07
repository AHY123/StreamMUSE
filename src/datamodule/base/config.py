from pydantic import BaseModel, Field
from ...utils.base_config import BaseConfig, Field, dataclass, ConfigDict
from typing import Any, Optional
from typing import Optional, Any, Union,Literal
import miditok


# Abstract (parent) dataset config
@dataclass
class BaseDatasetConfig(BaseConfig):
    _target_:Literal["src.datamodule.base.datamodule.BaseDataset"] = "src.datamodule.base.datamodule.BaseDataset"
    batch_size: int = Field(32, description="Batch size for data loaders.")
    num_workers: int = Field(4, description="Number of workers for data loading.")
    transform: Optional[Any] = Field(None, description="Optional transform to apply to data.")
    max_seq_len: Optional[int] = Field(384, description="Maximum sequence length for data. Default is 384.")
    data_range: Optional[tuple[float]] = Field(
        (0.0, 1.0), description="Optional range for data split (e.g., (0.0, 1.0) for full range). Default is (0.0, 1.0)."
    )
    stage: Optional[str] = Field("train", description="Stage of the dataset (e.g., 'train', 'val', 'test'). Optional if not needed.")


# Abstract (parent) datamodule config
@dataclass
class BaseDataModuleConfig(BaseConfig):
    _target_ : Literal["src.datamodule.base.datamodule.BaseDataModule"] = "src.datamodule.base.datamodule.BaseDataModule"
    train_config: Optional[BaseDatasetConfig] = Field(None, description="Configuration for the training dataset.")
    val_config: Optional[BaseDatasetConfig] = Field(None, description="Configuration for the validation dataset.")
    test_config: Optional[BaseDatasetConfig] = Field(None, description="Configuration for the test dataset.")
    predict_config: Optional[BaseDatasetConfig] = Field(None, description="Optional configuration for the prediction dataset.")

    def model_post_init(self, context: Any) -> None:
        if self.train_config:
            self.train_config.stage = "train"
        if self.val_config:
            self.val_config.stage = "val"
        if self.test_config:
            self.test_config.stage = "test"
        if self.predict_config:
            self.predict_config.stage = "predict"
