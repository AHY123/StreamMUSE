from pydantic import BaseModel, Field
from ...utils.base_config import BaseConfig, Field, dataclass, ConfigDict
from typing import Any, Optional
from typing import Optional, Any, Union,Literal
import miditok
from ..base.config import BaseDatasetConfig,BaseDataModuleConfig

@dataclass
class OldPtDatasetConfig(BaseDatasetConfig):
    _target_: Literal["src.datamodule.old_pt.datamodule.OldPtDataset"] = "src.datamodule.old_pt.datamodule.OldPtDataset"
    file_path: str = Field(default="")
    target_length: int = Field(default=384)
    split_ratio: int = Field(default=10)
    sequence_shift: int = Field(default=0)
    
    
@dataclass
class OldPtDataModuleConfig(BaseDataModuleConfig):
    _target_: Literal["src.datamodule.old_pt.datamodule.OldPtDataModule"] = "src.datamodule.old_pt.datamodule.OldPtDataModule"
    train_config: Optional[OldPtDatasetConfig] = Field(None, description="Configuration for the training dataset.")
    val_config: Optional[OldPtDatasetConfig] = Field(None, description="Configuration for the validation dataset.")
    test_config: Optional[OldPtDatasetConfig] = Field(None, description="Configuration for the test dataset.")
    predict_config: Optional[OldPtDatasetConfig] = Field(None,description="Configuration for the prediction dataset.")
    