from ..pl_base_model import PlBaseModelConfig
from ...network import CustomedRoformerConfig
from typing import Optional,Literal
from pydantic import Field
class OldM2ARoformerConfig(PlBaseModelConfig):
    _target_:Literal["src.model.specific_model.old_m2a_roformer.model.OldM2ATransformer"] = "src.model.specific_model.old_m2a_roformer.model.OldM2ATransformer"
    local_encoder_network_config: CustomedRoformerConfig = Field(default_factory=lambda: CustomedRoformerConfig())
    global_network_config: CustomedRoformerConfig = Field(default_factory=lambda: CustomedRoformerConfig())
    local_decoder_network_config: CustomedRoformerConfig = Field(default_factory=lambda: CustomedRoformerConfig())
    