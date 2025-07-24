from ..pl_base_model.config import PlBaseModelConfig, dataclass
from ...network import CustomedRoformerConfig
from ...tokenizer.base.config import BaseTokenizerConfig
from typing import Optional, Literal
from pydantic import Field


@dataclass
class PlTokenizerBaseModelConfig(PlBaseModelConfig):
    _target_: Literal["src.model.pl_tokenizer_base_model.model.FnsM2ATransformer"] = (
        "src.model.pl_tokenizer_base_model.model.FnsM2ATransformer"
    )
    tokenizer_config: BaseTokenizerConfig = Field(default_factory=lambda: BaseTokenizerConfig())
    # sub_seq_len:int =Field(4*3,description="4(polyphony) * 3(program,pitch,duration)*2(melody,accompany)")