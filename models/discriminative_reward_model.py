import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.roformer.modeling_roformer import RoFormerEncoder
from .base_reward_model import RewardModelBase
from schema.model_schema import DiscriminativeRewardModelSchema
from schema.model_io_schema import DiscriminativeBatch, RewardOutput
import time


class DiscriminativeRewardModel(RewardModelBase):
    """
    Discriminative reward model for local quality assessment.
    Uses single RoFormer encoder to classify melody-accompaniment combinations as real or fake.
    """
    
    def __init__(self, model_schema: DiscriminativeRewardModelSchema):
        super().__init__(model_schema)
        
        self.pooling_strategy = model_schema.pooling_strategy
        self.num_classes = model_schema.num_classes
        
        # Single RoFormer encoder for processing interleaved sequences
        self.encoder = RoFormerEncoder(self.roformer_config)
        
        # Embedding layer for combined melody+accompaniment tokens
        self.token_embedding = nn.Embedding(self.tokenizer_vocab_size, self.hidden_size)
        
        # CLS token embedding if using CLS pooling strategy
        if self.pooling_strategy == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))
            
        # Classification head
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        self.dropout = nn.Dropout(self.model_schema.hidden_dropout_prob)
        
    def create_interleaved_sequence(self, melody_tokens: torch.Tensor, accompaniment_tokens: torch.Tensor) -> torch.Tensor:
        """
        Create interleaved sequence from melody and accompaniment tokens.
        
        Args:
            melody_tokens: [seq_len] melody token sequence
            accompaniment_tokens: [seq_len] accompaniment token sequence
            
        Returns:
            torch.Tensor: [2*seq_len] interleaved sequence
        """
        # Simple interleaving strategy: alternate tokens
        interleaved = torch.stack([melody_tokens, accompaniment_tokens], dim=1).flatten()
        return interleaved
    
    def global_pooling(self, sequence_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply global pooling to sequence embeddings.
        
        Args:
            sequence_embeddings: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            torch.Tensor: [batch_size, hidden_size] pooled representation
        """
        if self.pooling_strategy == "cls":
            # Use first token (CLS) as global representation
            return sequence_embeddings[:, 0]  # [batch_size, hidden_size]
        
        elif self.pooling_strategy == "mean":
            # Mean pooling over non-padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_embeddings.size())
            masked_embeddings = sequence_embeddings * mask_expanded
            pooled = masked_embeddings.sum(1) / mask_expanded.sum(1)
            return pooled  # [batch_size, hidden_size]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
    
    def forward(self, batch: DiscriminativeBatch) -> torch.Tensor:
        """
        Forward pass for binary classification training.
        
        Args:
            batch: DiscriminativeBatch containing interleaved sequences and labels
            
        Returns:
            torch.Tensor: Binary cross-entropy loss
        """
        batch_size, seq_len = batch.interleaved_sequences.shape
        
        # Create token embeddings
        token_embeds = self.token_embedding(batch.interleaved_sequences)  # [B, seq_len, hidden_size]
        
        # Add CLS token if using CLS pooling
        if self.pooling_strategy == "cls":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, hidden_size]
            token_embeds = torch.cat([cls_tokens, token_embeds], dim=1)  # [B, seq_len+1, hidden_size]
            
            # Extend attention mask for CLS token
            cls_mask = torch.ones(batch_size, 1, device=batch.attention_masks.device)
            attention_mask = torch.cat([cls_mask, batch.attention_masks], dim=1)  # [B, seq_len+1]
        else:
            attention_mask = batch.attention_masks
        
        # Convert to 4D attention mask for RoFormer: [batch_size, 1, 1, seq_len]
        extended_attention_mask = attention_mask[:, None, None, :]
        # Convert 1s and 0s to 0s and -inf for masking
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Pass through encoder
        encoder_outputs = self.encoder(
            hidden_states=token_embeds,
            attention_mask=extended_attention_mask
        )
        
        # Apply global pooling
        sequence_output = encoder_outputs.last_hidden_state  # [B, seq_len(+1), hidden_size]
        pooled_output = self.global_pooling(sequence_output, attention_mask)  # [B, hidden_size]
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [B, num_classes]
        
        # Compute binary cross-entropy loss
        loss = F.cross_entropy(logits, batch.labels.long())
        
        return loss
    
    def predict_proba(self, interleaved_sequence: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Predict probabilities for single sequence.
        
        Args:
            interleaved_sequence: [seq_len] interleaved melody+accompaniment sequence
            attention_mask: [seq_len] attention mask (optional)
            
        Returns:
            torch.Tensor: [num_classes] probability distribution
        """
        self.eval()
        
        with torch.no_grad():
            # Add batch dimension
            seq_batch = interleaved_sequence.unsqueeze(0)  # [1, seq_len]
            
            if attention_mask is None:
                attention_mask = (seq_batch != 3204).float()  # PAD_TOKEN = 3204
            else:
                attention_mask = attention_mask.unsqueeze(0)  # [1, seq_len]
            
            # Create token embeddings
            token_embeds = self.token_embedding(seq_batch)  # [1, seq_len, hidden_size]
            
            # Add CLS token if using CLS pooling
            if self.pooling_strategy == "cls":
                cls_tokens = self.cls_token  # [1, 1, hidden_size]
                token_embeds = torch.cat([cls_tokens, token_embeds], dim=1)  # [1, seq_len+1, hidden_size]
                
                # Extend attention mask for CLS token
                cls_mask = torch.ones(1, 1, device=attention_mask.device)
                attention_mask = torch.cat([cls_mask, attention_mask], dim=1)  # [1, seq_len+1]
            
            # Convert to 4D attention mask for RoFormer: [batch_size, 1, 1, seq_len]
            extended_attention_mask = attention_mask[:, None, None, :]
            # Convert 1s and 0s to 0s and -inf for masking
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            
            # Pass through encoder
            encoder_outputs = self.encoder(
                hidden_states=token_embeds,
                attention_mask=extended_attention_mask
            )
            
            # Apply global pooling
            sequence_output = encoder_outputs.last_hidden_state
            pooled_output = self.global_pooling(sequence_output, attention_mask)  # [1, hidden_size]
            
            # Get logits and probabilities
            logits = self.classifier(pooled_output)  # [1, num_classes]
            probabilities = F.softmax(logits, dim=-1)  # [1, num_classes]
            
            return probabilities.squeeze(0)  # [num_classes]
    
    def compute_reward(self, melody_tokens: torch.Tensor, accompaniment_tokens: torch.Tensor) -> float:
        """
        Compute reward score for single melody-accompaniment pair.
        
        Args:
            melody_tokens: [seq_len] melody token sequence
            accompaniment_tokens: [seq_len] accompaniment token sequence
            
        Returns:
            float: P(real) probability (0 to 1)
        """
        start_time = time.time()
        
        # Create interleaved sequence
        interleaved_seq = self.create_interleaved_sequence(melody_tokens, accompaniment_tokens)
        
        # Get probability distribution
        probabilities = self.predict_proba(interleaved_seq)
        
        # Return P(real) - probability of positive class (index 1)
        p_real = probabilities[1].item()
        
        computation_time = time.time() - start_time
        
        return RewardOutput(
            reward_score=p_real,
            model_type="Discriminative-Reward-Model",
            sequence_length=len(interleaved_seq),
            computation_time=computation_time
        ).reward_score
    
    def training_step(self, batch, batch_idx):
        """Training step for PyTorch Lightning."""
        loss = self.forward(batch)
        
        # Compute accuracy for monitoring
        with torch.no_grad():
            batch_size, seq_len = batch.interleaved_sequences.shape
            
            # Get predictions
            token_embeds = self.token_embedding(batch.interleaved_sequences)
            
            if self.pooling_strategy == "cls":
                cls_tokens = self.cls_token.expand(batch_size, -1, -1)
                token_embeds = torch.cat([cls_tokens, token_embeds], dim=1)
                cls_mask = torch.ones(batch_size, 1, device=batch.attention_masks.device)
                attention_mask = torch.cat([cls_mask, batch.attention_masks], dim=1)
            else:
                attention_mask = batch.attention_masks
            
            encoder_outputs = self.encoder(hidden_states=token_embeds, attention_mask=attention_mask)
            pooled_output = self.global_pooling(encoder_outputs.last_hidden_state, attention_mask)
            logits = self.classifier(pooled_output)
            
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == batch.labels).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', accuracy, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step with accuracy computation."""
        loss = self.forward(batch)
        
        # Compute accuracy
        with torch.no_grad():
            batch_size, seq_len = batch.interleaved_sequences.shape
            
            token_embeds = self.token_embedding(batch.interleaved_sequences)
            
            if self.pooling_strategy == "cls":
                cls_tokens = self.cls_token.expand(batch_size, -1, -1)
                token_embeds = torch.cat([cls_tokens, token_embeds], dim=1)
                cls_mask = torch.ones(batch_size, 1, device=batch.attention_masks.device)
                attention_mask = torch.cat([cls_mask, batch.attention_masks], dim=1)
            else:
                attention_mask = batch.attention_masks
            
            encoder_outputs = self.encoder(hidden_states=token_embeds, attention_mask=attention_mask)
            pooled_output = self.global_pooling(encoder_outputs.last_hidden_state, attention_mask)
            logits = self.classifier(pooled_output)
            
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == batch.labels).float().mean()
        
        # Log validation metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', accuracy, prog_bar=True)
        
        return loss