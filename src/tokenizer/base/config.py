from miditok import  TokenizerConfig
from pydantic.dataclasses import dataclass
from typing import Optional, Literal
from pydantic import Field,model_validator,ConfigDict

@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class BaseTokenizerConfig:
    """
    BaseTokenizerConfig is a configuration class for the BaseTokenizerConfig.
    It inherits from miditok.TokenizerConfig and can be used to customize the tokenizer's behavior.
    """
    _target_: Literal["src.tokenizer.base.tokenizer.BaseTokenizerConfig"] = "src.tokenizer.base.tokenizer.BaseTokenizerConfig"
    config : TokenizerConfig = Field(TokenizerConfig())
    
    @model_validator(mode="after")
    def validate_config(self)->"BaseTokenizerConfig":
        self.config.one_token_stream_for_programs=True
        # self.config.use_programs = True
        return self
    

