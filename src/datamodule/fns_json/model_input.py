from ...utils.base_config import dataclass, Field
from pydantic import ConfigDict
from typing import Optional, Union
import torch
from miditok import TokSequence

@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class FnsInput:
    # batched_tokens: Optional[Union[TokSequence,list[TokSequence]]]= Field(
    #     default=None,
    #     description="Batched tokens, either as a TokSequence or a list of TokSequences."
    # )
    
    token_ids: Optional[Union[torch.Tensor,list[torch.Tensor]]] = Field(
        default=None,
        description="Batched token IDs, represented as a PyTorch tensor."
    )
