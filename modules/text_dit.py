# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import logging
from diffusers.models.attention import BasicTransformerBlock
# We will NOT use diffusers.models.embeddings.PatchEmbed
# from diffusers.models.embeddings import PatchEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput # Keep for now, but conceptualize as TransformerTextModelOutput
from diffusers.models.modeling_utils import ModelMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Define a new input embedding layer for text, similar to PatchEmbed but for tokens
class TextTokenEmbed(nn.Module):
    """
    TextTokenEmbed takes discrete token IDs and converts them into continuous embeddings
    suitable for the Transformer. It also adds positional embeddings.
    """
    def __init__(self, vocab_size: int, max_seq_len: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        # Token embedding layer (replaces image patch embedding)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # Positional embedding layer
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, token_ids: torch.Tensor):
        # token_ids: (batch_size, sequence_length)
        batch_size, seq_len = token_ids.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds maximum sequence length ({self.max_seq_len})"
            )

        token_embeds = self.token_embedding(token_ids) # (batch_size, seq_len, embed_dim)

        # Generate positional IDs: (0, 1, ..., seq_len-1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=token_ids.device)
        position_embeds = self.position_embedding(position_ids) # (seq_len, embed_dim)

        # Add positional embeddings to token embeddings
        # unsqueeze(0) to broadcast position_embeds to all items in the batch
        embeddings = token_embeds + position_embeds.unsqueeze(0)
        return embeddings


class DiTTransformerTextModel(ModelMixin, ConfigMixin):
    r"""
    A Transformer model adapted from DiT for text generation, suitable for discrete diffusion processes.

    Parameters:
        vocab_size (int): The size of the vocabulary.
        max_seq_len (int): The maximum sequence length of the text.
        num_attention_heads (int, optional, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (int, optional, defaults to 72): The number of channels in each head.
        hidden_size (int, defaults to 768): The dimension of the Transformer's hidden states (and token embeddings).
        num_layers (int, optional, defaults to 28): The number of layers of Transformer blocks to use.
        dropout (float, optional, defaults to 0.0): The dropout probability to use within the Transformer blocks.
        norm_num_groups (int, optional, defaults to 32):
            Number of groups for group normalization within Transformer blocks (might not be used directly in AdaNorm).
        attention_bias (bool, optional, defaults to True):
            Configure if the Transformer blocks' attention should contain a bias parameter.
        activation_fn (str, optional, defaults to "gelu-approximate"):
            Activation function to use in feed-forward networks within Transformer blocks.
        num_embeds_ada_norm (int, optional, defaults to 1000):
            Number of embeddings for AdaLayerNorm, fixed during training and affects the maximum denoising steps during
            inference.
        upcast_attention (bool, optional, defaults to False):
            If true, upcasts the attention mechanism dimensions for potentially improved performance.
        norm_type (str, optional, defaults to "ada_norm_zero"):
            Specifies the type of normalization used, can be 'ada_norm_zero'.
        norm_elementwise_affine (bool, optional, defaults to False):
            If true, enables element-wise affine parameters in the normalization layers.
        norm_eps (float, optional, defaults to 1e-5):
            A small constant added to the denominator in normalization layers to prevent division by zero.
    """

    _skip_layerwise_casting_patterns = ["pos_embed", "norm"] # Keep for consistency, but pos_embed is now inside TextTokenEmbed
    _supports_gradient_checkpointing = True
    _supports_group_offloading = False

    @register_to_config
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        num_attention_heads: int = 16,
        attention_head_dim: int = 72,
        hidden_size: int = 768, # Represents the embedding dimension and inner_dim
        num_layers: int = 28,
        dropout: float = 0.0,
        norm_num_groups: int = 32, # May not be strictly necessary with AdaNorm
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm_zero",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        # Validate inputs.
        if norm_type != "ada_norm_zero":
            raise NotImplementedError(
                f"Forward pass is not implemented when `norm_type` is '{norm_type}' for text models."
            )
        elif norm_type == "ada_norm_zero" and num_embeds_ada_norm is None:
            raise ValueError(
                f"When using this `norm_type` ({norm_type}), `num_embeds_ada_norm` cannot be None."
            )

        # Set some common variables used across the board.
        self.attention_head_dim = attention_head_dim
        self.inner_dim = hidden_size # Transformer's internal dimension
        self.gradient_checkpointing = False

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # 1. Initialize the token and position embedding.
        # Replaced PatchEmbed with our custom TextTokenEmbed
        self.text_embed = TextTokenEmbed(
            vocab_size=self.config.vocab_size,
            max_seq_len=self.config.max_seq_len,
            embed_dim=self.inner_dim,
        )

        # 2. Transformer blocks.
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        # 3. Output blocks.
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        # Modified proj_out_2: It now projects to vocab_size for token logits
        self.proj_out_2 = nn.Linear(self.inner_dim, self.vocab_size)

    def forward(
        self,
        hidden_states: torch.Tensor, # This will now be token IDs (batch_size, seq_len)
        timestep: Optional[torch.LongTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        return_dict: bool = True,
    ):
        """
        The [`DiTTransformerTextModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, sequence_length)`):
                Input `hidden_states` representing discrete token IDs. These would be the 'noisy' text tokens.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple. (Will return Transformer2DModelOutput for now, but conceptually it's a TextOutput)

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor (representing token logits).
        """
        # 1. Input: Convert token IDs to embeddings
        # hidden_states is (batch_size, sequence_length)
        hidden_states = self.text_embed(hidden_states) # (batch_size, sequence_length, inner_dim)

        # 2. Blocks
        # The BasicTransformerBlock internally handles the AdaLayerNorm conditioning
        # using timestep and class_labels.
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    None, # attention_mask, would need to be passed if using
                    None, # encoder_hidden_states
                    None, # encoder_attention_mask
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None, # You might want to add a proper attention mask here for padding
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                )

        # 3. Output
        # Get conditioning for output normalization from the first transformer block's norm1 (AdaLayerNormZero)
        # Note: This assumes the first block's norm1 is an AdaLayerNormZero, which is typical for DiT
        conditioning = self.transformer_blocks[0].norm1.emb(timestep, class_labels, hidden_dtype=hidden_states.dtype)
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]

        # Project to vocabulary size to get logits for each token
        # hidden_states is (batch_size, sequence_length, inner_dim)
        # proj_out_2 maps (inner_dim) -> (vocab_size)
        output_logits = self.proj_out_2(hidden_states) # (batch_size, sequence_length, vocab_size)

        # The output `output_logits` now directly represents the predicted logits
        # for each token in the sequence.
        # This replaces the complex `unpatchify` and `einsum` operations from the image DiT.

        if not return_dict:
            return (output_logits,)

        # We're reusing Transformer2DModelOutput, but conceptually this is a TextModelOutput
        return Transformer2DModelOutput(sample=output_logits)