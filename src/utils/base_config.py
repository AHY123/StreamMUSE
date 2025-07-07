from pydantic.dataclasses import dataclass
from pydantic import Field,ConfigDict


@dataclass
class BaseConfig:
    _target_: str = Field("src.utils.base_config.BaseConfig")
