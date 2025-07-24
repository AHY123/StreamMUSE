from ..pl_tokenzier_base_model.config import PlTokenizerBaseModelConfig, dataclass
from ...network import CustomedRoformerConfig
from ...tokenizer.fns.config import FnsTokenizerConfig
from typing import Optional, Literal
from pydantic import Field


@dataclass
class FnsM2ARoformerConfig(PlTokenizerBaseModelConfig):
    _target_: Literal["src.model.fns_m2a_roformer.model.FnsM2ATransformer"] = (
        "src.model.fns_m2a_roformer.model.FnsM2ATransformer"
    )
    local_encoder_network_config: CustomedRoformerConfig = Field(default_factory=lambda: CustomedRoformerConfig())
    global_network_config: CustomedRoformerConfig = Field(default_factory=lambda: CustomedRoformerConfig())
    local_decoder_network_config: CustomedRoformerConfig = Field(default_factory=lambda: CustomedRoformerConfig())
    tokenizer_config: FnsTokenizerConfig = Field(default_factory=lambda: FnsTokenizerConfig())
    sub_seq_len:int =Field(4*3,description="4(polyphony) * 3(program,pitch,duration)*2(melody,accompany)")