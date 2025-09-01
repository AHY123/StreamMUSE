# defaults:
#   - _self_
#   - /network/roformer/tiny@local_encoder_network_config  # 使用 tiny 模型配置
#   - /network/roformer/medium@global_network_config  # 使用 medium 模型配置
#   - /network/roformer/tiny@local_decoder_network_config  # 使用 tiny 模型配置

# _target_: src.model.old_m2a_roformer.model.OldM2ATransformer
# sub_seq_len: 384

from dataclasses import replace
from src.model.old_m2a_roformer.config import OldM2ARoformerConfig
from src.network.customed_roformer_network import CustomedRoformerConfig, RoFormerConfig
from conf_py.network.roformer.large import config as large_roformer_config
from conf_py.network.roformer.medium import config as medium_roformer_config
from conf_py.network.roformer.tiny import config as tiny_roformer_config

config = OldM2ARoformerConfig(
    local_encoder_network_config=tiny_roformer_config,
    global_network_config=medium_roformer_config,
    local_decoder_network_config=tiny_roformer_config,
    sub_seq_len=384,
)
