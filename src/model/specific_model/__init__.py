from .pl_base_model.model_io import PlBaseModelInput, PlBaseModelOutput
from .pl_base_model.config import PlBaseModelConfig
from .old_m2a_roformer.model_io import OldM2ARoformerInput, OldM2ARoformerOutput
from .old_m2a_roformer.config import OldM2ARoformerConfig
from typing import Union, Optional, Any

UnionModelInput = Union[PlBaseModelInput, OldM2ARoformerInput]
UnionModelOutput = Union[PlBaseModelOutput, OldM2ARoformerOutput]
UnionModelConfig = Union[PlBaseModelConfig, OldM2ARoformerConfig]