# # @package network_roformer
# defaults:
#   - _self_

# _target_: src.network.customed_roformer_network.CustomedRoformerNetwork
# vocab_size: 50000
# embedding_size: null
# hidden_size: 768
# hidden_act: "gelu"
# hidden_dropout_prob: 0.1
# attention_probs_dropout_prob: 0.1
# max_position_embeddings: 1536
# type_vocab_size: 2
# initializer_range: 0.02
# layer_norm_eps: 1.0e-12
# pad_token_id: 0

from src.network.customed_roformer_network import CustomedRoformerConfig, RoFormerConfig

config = CustomedRoformerConfig(
    config=RoFormerConfig(
        vocab_size=1200,
        hidden_size=768,
        hidden_act="gelu",
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0,
        type_vocab_size=2,
        max_position_embeddings=1536,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1.0e-12,
        pad_token_id=0,
    )
)
