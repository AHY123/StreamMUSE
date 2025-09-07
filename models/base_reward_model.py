import torch
from abc import ABC, abstractmethod
from transformers.models.roformer.modeling_roformer import RoFormerConfig
from schema.model_schema import ContrastiveRewardModelSchema, DiscriminativeRewardModelSchema
from .base_pytorch_lightning_model import BasePyTorchLightningModel
from typing import Union, Tuple


class RewardModelBase(BasePyTorchLightningModel, ABC):
    """
    Abstract base class for reward models providing common functionality.
    Inherits from BasePyTorchLightningModel to maintain compatibility with existing infrastructure.
    """
    
    def __init__(self, model_schema: Union[ContrastiveRewardModelSchema, DiscriminativeRewardModelSchema]):
        super().__init__(model_schema)
        self.model_schema = model_schema
        self.hidden_size = model_schema.hidden_size
        self.num_layers = model_schema.num_layers
        self.num_attention_heads = model_schema.num_attention_heads
        self.intermediate_size = model_schema.intermediate_size
        
        # Tokenizer will be set up by child classes based on StreamMUSE tokenization
        self.tokenizer_vocab_size = 3205  # N_TOKENS from StreamMUSE (3202 normal + SOS/EOS/PAD)
        
        # Configure RoFormer for consistency with StreamMUSE
        self.roformer_config = self._create_roformer_config()
    
    def _create_roformer_config(self) -> RoFormerConfig:
        """Create RoFormer configuration compatible with StreamMUSE."""
        return RoFormerConfig(
            vocab_size=self.tokenizer_vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=self.model_schema.hidden_dropout_prob,
            attention_probs_dropout_prob=self.model_schema.attention_probs_dropout_prob,
            max_position_embeddings=1024,  # Sufficient for expected sequence lengths
            layer_norm_eps=1e-12,
            initializer_range=0.02,
            use_cache=False,  # Not needed for reward models
        )
    
    @abstractmethod
    def compute_reward(self, melody_tokens: torch.Tensor, accompaniment_tokens: torch.Tensor) -> float:
        """
        Compute reward score for melody-accompaniment pair.
        
        Args:
            melody_tokens: Tokenized melody sequence [seq_len]
            accompaniment_tokens: Tokenized accompaniment sequence [seq_len]
        
        Returns:
            float: Reward score (range depends on model type)
        """
        pass
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass - implemented by child classes."""
        pass
    
    def configure_optimizers(self):
        """Configure optimizer following existing pattern."""
        optimizer_config = self.model_schema.optimizer_schema
        if optimizer_config.optimizer_type == "adam":
            optimizer = torch.optim.Adam(self.parameters(), **optimizer_config.params)
        elif optimizer_config.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), **optimizer_config.params)
        elif optimizer_config.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), **optimizer_config.params)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_config.optimizer_type}")
        
        return optimizer
    
    def validation_step(self, batch, batch_idx):
        """Common validation logic - can be overridden by child classes."""
        loss = self.training_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss