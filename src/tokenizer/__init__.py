from typing import Union

from .base.tokenizer import BaseTokenizer
from .base.config import BaseTokenizerConfig

from .fns.tokenizer import FnsTokenizer
from .fns.config import FnsTokenizerConfig

UnionTokenizerConfig = Union[BaseTokenizerConfig,FnsTokenizerConfig]  # Add other tokenizer configs as needed
UnionTokenizer = Union[BaseTokenizerConfig,FnsTokenizer]  # Add other tokenizers as needed
