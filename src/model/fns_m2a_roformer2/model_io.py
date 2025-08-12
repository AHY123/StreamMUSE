from ..pl_base_model.model_io import PlBaseModelInput, PlBaseModelOutput
from ...datamodule.fns_json.model_input import FnsInput as FnsM2ARoformer2Input
from ...utils.base_config import dataclass, Field
from pydantic import ConfigDict
from typing import Optional, Union
import torch
from miditok import TokSequence

# @dataclass(config=ConfigDict(arbitrary_types_allowed=True))
# class FnsM2ARoformer2Input:
#     batched_tokens: Union[TokSequence,list[TokSequence]]= Field(
#         default=None,
#         description="Batched tokens, either as a TokSequence or a list of TokSequences."
#     )


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class FnsM2ARoformerOutput(PlBaseModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = Field(
        default=None,
        description="Sequence of hidden-states at the output of the last layer of the model.",
    )
