from pydantic import BaseModel, Field
from ...utils.base_config import BaseConfig, Field, dataclass, ConfigDict
from typing import Any, Optional
from typing import Optional, Any, Union,Literal
import miditok
from ..base.config import BaseDatasetConfig,BaseDataModuleConfig


class OldPtDatasetConfig(BaseDatasetConfig):
    _target_: Literal["src.datamodule.old_pt.datamodule.OldPtDataset"] = "src.datamodule.old_pt.datamodule.OldPtDataset"
    ...
    
class OldPtDataModuleConfig(BaseDataModuleConfig):
    _target_: Literal["src.datamodule.old_pt.datamodule.OldPtDataModule"] = "src.datamodule.old_pt.datamodule.OldPtDataModule"
    train_config: Optional[OldPtDatasetConfig] = Field(None, description="Configuration for the training dataset.")
    val_config: Optional[OldPtDatasetConfig] = Field(None, description="Configuration for the validation dataset.")
    test_config: Optional[OldPtDatasetConfig] = Field(None, description="Configuration for the test dataset.")
    predict_config: Optional[OldPtDatasetConfig] = Field()
    