from ..pl_base_model.model_io import PlBaseModelInput,PlBaseModelOutput
from ....utils.base_config import dataclass, Field
from typing import Optional,Union
import torch

@dataclass
class OldM2ARoformerInput(PlBaseModelInput):
    token_ids: Optional[torch.Tensor] = Field(
        default=None,
        description="Indices of input sequence tokens in the vocabulary.",
        
    )
    mel_data: Optional[torch.Tensor] = Field(
        default=None,
        description="Melody data.",
    )
    acc_data: Optional[torch.Tensor] = Field(
        default=None,
        description="Accompaniment data.",
    )
    pitch_shift: Optional[torch.Tensor] = Field(
        default=None,
        description="Pitch shift data.",
    )

@dataclass
class OldM2ARoformerOutput(PlBaseModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = Field(
        default=None,
        description="Sequence of hidden-states at the output of the last layer of the model.",
    )
    
    