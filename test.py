from pydantic.dataclasses import dataclass
from pydantic import Field
from hydra import compose, initialize
from typing import Optional


@dataclass
class BaseConfig:
    model_name: str = Field(default="")

@dataclass
class ConfigMixin:
    _config:BaseConfig = Field(default_factory=BaseConfig)
    
class Model(ConfigMixin):
    def __init__(self, config: Optional[BaseConfig] = None, **kwargs):
        if config is None:
            config = BaseConfig(**kwargs)
        self._config = config
        super().__init__(_config=self._config)
        self.a = 1
        
if __name__ == "__main__":
    config = BaseConfig(model_name="test_model")
    model = Model(config)
    print(model._config)
    print(model)
    print(model.a)
    
    model2 = Model(model_name="test_model")
    print(model2._config)
    print(model2)
    print(model2.a)
    