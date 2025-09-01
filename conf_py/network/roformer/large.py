# defaults:
#   - /network/roformer/base@_here_
#   - _self_
# num_hidden_layers: 12
# num_attention_heads: 12
# intermediate_size: 3072
from src.network.customed_roformer_network import CustomedRoformerConfig, RoFormerConfig
from .base import config as base_config
from dataclasses import replace

# 1. 从 base_config 中获取底层的 RoFormerConfig
base_roformer_conf = base_config.config

# 2. 使用 replace 创建一个新的 RoFormerConfig 实例，并覆盖指定参数
#    这完全模仿了 Hydra 的行为：继承 base，然后应用 self 的值。
large_roformer_conf = replace(
    base_roformer_conf,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
)

# 3. 用新的、被覆盖后的配置来创建最终的 config 对象
config = CustomedRoformerConfig(config=large_roformer_conf)


# ============== Alternative Approach ==============
# from dataclasses import asdict
# from src.network.customed_roformer_network import CustomedRoformerConfig, RoFormerConfig
# from .base import config as base_config

# # 1. 将基础配置转换为字典
# base_params = asdict(base_config.config)

# # 2. 定义需要覆盖的参数
# overrides = {"num_hidden_layers": 12, "num_attention_heads": 12, "intermediate_size": 3072}

# # 3. 合并字典，overrides 中的值会覆盖 base_params 中的值
# final_params = {**base_params, **overrides}

# # 4. 使用最终的参数字典创建新的配置实例
# config = CustomedRoformerConfig(config=RoFormerConfig(**final_params))
