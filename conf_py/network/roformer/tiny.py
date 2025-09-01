# defaults:
#   - /network/roformer/base@_here_
#   - _self_
# num_hidden_layers: 4
# num_attention_heads: 4
# intermediate_size: 1024

from src.network.customed_roformer_network import CustomedRoformerConfig, RoFormerConfig
from .base import config as base_config
from dataclasses import replace

base_roformer_conf = base_config.config
tiny_roformer_conf = replace(
    base_roformer_conf,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=1024,
)
config = CustomedRoformerConfig(config=tiny_roformer_conf)