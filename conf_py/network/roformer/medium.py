# defaults:
#   - /network/roformer/base@_here_
#   - _self_
# num_hidden_layers: 6
# num_attention_heads: 6
# intermediate_size: 1536

from src.network.customed_roformer_network import CustomedRoformerConfig, RoFormerConfig
from .base import config as base_config
from dataclasses import replace

base_roformer_conf = base_config.config
medium_roformer_conf = replace(
    base_roformer_conf,
    num_hidden_layers=6,
    num_attention_heads=6,
    intermediate_size=1536,
)
config = CustomedRoformerConfig(config=medium_roformer_conf)
