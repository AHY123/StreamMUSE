import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.roformer.modeling_roformer import RoFormerEncoder
from .base_reward_model import RewardModelBase
from schema.model_schema import ContrastiveRewardModelSchema
from schema.model_io_schema import ContrastiveBatch, RewardOutput
import time
from typing import Tuple


class ContrastiveRewardModel(RewardModelBase):
    """
    Contrastive reward model for global harmony assessment.
    Uses dual RoFormer encoders to learn melody-accompaniment relationships.
    """
    
    def __init__(self, model_schema: ContrastiveRewardModelSchema):
        super().__init__(model_schema)
        
        self.temperature = model_schema.temperature
        self.embedding_dim = model_schema.embedding_dim
        
        # Create dual RoFormer encoders
        self.melody_encoder = RoFormerEncoder(self.roformer_config)
        self.accompaniment_encoder = RoFormerEncoder(self.roformer_config)
        
        # Embedding layers for melody and accompaniment tokens
        self.melody_embedding = nn.Embedding(self.tokenizer_vocab_size, self.hidden_size)
        self.accompaniment_embedding = nn.Embedding(self.tokenizer_vocab_size, self.hidden_size)
        
        # Projection layers to map encoder outputs to embedding space
        self.melody_projection = nn.Linear(self.hidden_size, self.embedding_dim)
        self.accompaniment_projection = nn.Linear(self.hidden_size, self.embedding_dim)
        
    def encode_melody(self, melody_tokens: torch.Tensor) -> torch.Tensor:
        """
        Encode melody sequence to embedding vector.
        
        Args:
            melody_tokens: [batch_size, seq_len]
            
        Returns:
            torch.Tensor: [batch_size, embedding_dim]
        """
        # Create embeddings
        melody_embeds = self.melody_embedding(melody_tokens)  # [B, seq_len, hidden_size]
        
        # Create attention mask (non-padding tokens)
        attention_mask = (melody_tokens != 3204).float()  # PAD_TOKEN = 3204
        
        # Pass through encoder
        encoder_outputs = self.melody_encoder(
            hidden_states=melody_embeds,
            attention_mask=attention_mask
        )
        
        # Global pooling - use mean pooling over non-padding tokens
        sequence_output = encoder_outputs.last_hidden_state  # [B, seq_len, hidden_size]
        mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size())
        masked_output = sequence_output * mask_expanded
        pooled_output = masked_output.sum(1) / mask_expanded.sum(1)  # [B, hidden_size]
        
        # Project to embedding space
        melody_embedding = self.melody_projection(pooled_output)  # [B, embedding_dim]
        
        return F.normalize(melody_embedding, dim=-1)  # L2 normalize for cosine similarity
    
    def encode_accompaniment(self, accompaniment_tokens: torch.Tensor) -> torch.Tensor:
        """
        Encode accompaniment sequence to embedding vector.
        
        Args:
            accompaniment_tokens: [batch_size, seq_len]
            
        Returns:
            torch.Tensor: [batch_size, embedding_dim]
        """
        # Create embeddings
        acc_embeds = self.accompaniment_embedding(accompaniment_tokens)  # [B, seq_len, hidden_size]
        
        # Create attention mask (non-padding tokens)
        attention_mask = (accompaniment_tokens != 3204).float()  # PAD_TOKEN = 3204
        
        # Pass through encoder
        encoder_outputs = self.accompaniment_encoder(
            hidden_states=acc_embeds,
            attention_mask=attention_mask
        )
        
        # Global pooling - use mean pooling over non-padding tokens
        sequence_output = encoder_outputs.last_hidden_state  # [B, seq_len, hidden_size]
        mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size())
        masked_output = sequence_output * mask_expanded
        pooled_output = masked_output.sum(1) / mask_expanded.sum(1)  # [B, hidden_size]
        
        # Project to embedding space
        acc_embedding = self.accompaniment_projection(pooled_output)  # [B, embedding_dim]
        
        return F.normalize(acc_embedding, dim=-1)  # L2 normalize for cosine similarity
    
    def compute_similarity_matrix(self, melody_embeddings: torch.Tensor, acc_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity matrix between all melody-accompaniment pairs in batch.
        
        Args:
            melody_embeddings: [B, embedding_dim]
            acc_embeddings: [B, embedding_dim]
            
        Returns:
            torch.Tensor: [B, B] similarity matrix
        """
        # Both embeddings are already L2-normalized in encode methods
        # So matrix multiplication gives cosine similarity
        similarity_matrix = torch.mm(melody_embeddings, acc_embeddings.t())
        return similarity_matrix
    
    def info_nce_loss(self, similarity_matrix: torch.Tensor, positive_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute InfoNCE loss from similarity matrix.
        
        Args:
            similarity_matrix: [B, B] cosine similarity matrix
            positive_mask: [B, B] mask for positive pairs (optional)
            
        Returns:
            torch.Tensor: InfoNCE loss value
        """
        # Apply temperature scaling
        logits = similarity_matrix / self.temperature
        
        # For contrastive learning, assume diagonal elements are positive pairs
        if positive_mask is None:
            labels = torch.arange(logits.size(0), device=logits.device)
            loss = F.cross_entropy(logits, labels)
        else:
            # More general InfoNCE with custom positive pairs
            # This implementation assumes single positive per query for simplicity
            labels = torch.arange(logits.size(0), device=logits.device)
            loss = F.cross_entropy(logits, labels)
            
        return loss

    def forward(self, batch: ContrastiveBatch) -> torch.Tensor:
        """
        Forward pass for training with InfoNCE loss.
        
        Args:
            batch: ContrastiveBatch containing melody/accompaniment pairs and labels
            
        Returns:
            torch.Tensor: InfoNCE loss value
        """
        # Encode sequences
        melody_embeddings = self.encode_melody(batch.melody_sequences)  # [B, embedding_dim]
        acc_embeddings = self.encode_accompaniment(batch.accompaniment_sequences)  # [B, embedding_dim]
        
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(melody_embeddings, acc_embeddings)  # [B, B]
        
        # Compute InfoNCE loss
        loss = self.info_nce_loss(similarity_matrix)
        
        return loss
    
    def compute_reward(self, melody_tokens: torch.Tensor, accompaniment_tokens: torch.Tensor) -> float:
        """
        Compute reward score for single melody-accompaniment pair.
        
        Args:
            melody_tokens: [seq_len]
            accompaniment_tokens: [seq_len]
            
        Returns:
            float: Cosine similarity score (-1 to +1)
        """
        self.eval()
        start_time = time.time()
        
        with torch.no_grad():
            # Add batch dimension
            melody_batch = melody_tokens.unsqueeze(0)  # [1, seq_len]
            acc_batch = accompaniment_tokens.unsqueeze(0)  # [1, seq_len]
            
            # Encode sequences
            melody_embedding = self.encode_melody(melody_batch)  # [1, embedding_dim]
            acc_embedding = self.encode_accompaniment(acc_batch)  # [1, embedding_dim]
            
            # Compute cosine similarity (already normalized in encode methods)
            similarity = torch.mm(melody_embedding, acc_embedding.t()).item()
            
        computation_time = time.time() - start_time
        
        return RewardOutput(
            reward_score=similarity,
            model_type="Contrastive-Reward-Model",
            sequence_length=len(melody_tokens),
            computation_time=computation_time
        ).reward_score
    
    def training_step(self, batch, batch_idx):
        """Training step for PyTorch Lightning."""
        loss = self.forward(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss