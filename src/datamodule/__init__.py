from .base.config import BaseDatasetConfig,BaseDataModuleConfig
from .old_pt.config import OldPtDatasetConfig,OldPtDataModuleConfig
from typing import Union

UnionDatasetConfig = Union[BaseDatasetConfig,OldPtDatasetConfig]
UnionDataModuleConfig = Union[BaseDataModuleConfig,OldPtDataModuleConfig]
