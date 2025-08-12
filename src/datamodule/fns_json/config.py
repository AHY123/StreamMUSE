from ..base.config import BaseDatasetConfig,BaseDataModuleConfig,dataclass
from ...tokenizer.fns.config import FnsTokenizerConfig
from pydantic import Field, model_validator
from typing import Optional, Union,Literal
from miditok.pytorch_data import DatasetJSON, DatasetMIDI

@dataclass
class FnsJsonDatasetConfig(BaseDatasetConfig):
    _target_: Literal["src.datamodule.fns_json.datamodule.FnsJsonDataset"] = "src.datamodule.fns_json.datamodule.FnsJsonDataset"
    # target_length: int = Field(default=1024, description="The target length of the sequences in the dataset.")
    max_frame_length: int = Field(default=192, description="The maximum length of the frames in the dataset.")
    json_dir: str = Field(default="", description="Directory containing the JSON files for the dataset.")
    tokenizer_config: Optional[FnsTokenizerConfig] = Field(
        default=FnsTokenizerConfig(), description="Configuration for the tokenizer used in the dataset."
    )

    @model_validator(mode="after")
    def validate_stage(self):
        if self.stage not in ["train", "val", "test", "all", "predict"]:
            raise ValueError(
                f"Invalid stage '{self.stage}'. Must be one of 'train', 'val', 'test', 'predict', or 'all'."
            )
        return self
    
@dataclass
class FnsJsonDataModuleConfig(BaseDataModuleConfig):
    _target_:Literal["src.datamodule.fns_json.datamodule.FnsJsonDataModule"] = "src.datamodule.fns_json.datamodule.FnsJsonDataModule"
    train_config: Optional[FnsJsonDatasetConfig] = Field(
        default=None, description="Configuration for the training dataset."
    )
    val_config: Optional[FnsJsonDatasetConfig] = Field(
        default=None, description="Configuration for the validation dataset."
    )
    test_config: Optional[FnsJsonDatasetConfig] = Field(
        default=None, description="Configuration for the test dataset."
    )   
    predict_config: Optional[FnsJsonDatasetConfig] = Field(
        default=None, description="Configuration for the prediction dataset."
    )

