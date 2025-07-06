from typing import Union, Optional, Any
from .customed_roformer_network import CustomedRoformerNetwork, CustomedRoformerConfig

UnionNetworkConfig = Union[CustomedRoformerConfig, Any]
UnionNetwork = Union[CustomedRoformerNetwork, Any]
