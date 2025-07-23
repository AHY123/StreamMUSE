from ....utils.base_config import dataclass, Field
from pydantic import ConfigDict
from typing import Optional
import torch


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class PlBaseModelInput:...
    # input_ids: Optional[torch.LongTensor] = Field(
    #     default=None,
    #     description="Indices of input sequence tokens in the vocabulary.",
    # )
    
    

@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class PlBaseModelOutput:...
    # last_hidden_state: torch.FloatTensor = Field(
    #     default=None,
    #     description="Sequence of hidden-states at the output of the last layer of the model.",
    # )