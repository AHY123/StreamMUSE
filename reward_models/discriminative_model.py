import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers.models.roformer.modeling_roformer import RoFormerEncoder, RoFormerConfig

# Duration templates from preprocessing pipeline (same as main model)
DURATION_TEMPLATES = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096])


class DiscriminativeRewardModel(nn.Module):
    """
    Discriminative reward model for binary classification of melody-accompaniment quality.
    
    Architecture mirrors the main StreamMUSE model:
    - Local encoder: Process polyphonic frames [seq_len, 12] -> frame representations
    - Global encoder: Process frame sequence -> final sequence embedding  
    - Classification head: Binary real/fake prediction
    """
    
    def __init__(self, hidden_size=512, num_layers=6, num_heads=8, vocab_size=3205):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Token constants (same as main model)
        self.SOS_TOKEN = 3202
        self.EOS_TOKEN = 3203  
        self.PAD_TOKEN = 3204
        
        # Local encoder configuration (same as main model)
        self.local_config = RoFormerConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=2,  # Local encoder is shallow
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=512,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        
        # Global encoder configuration  
        self.global_config = RoFormerConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=512,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        
        # Embedding layers (same as main model)
        self.local_embedding = nn.Embedding(vocab_size, hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)  # melody vs accompaniment
        
        # Encoder layers
        self.local_encoder = RoFormerEncoder(self.local_config)
        self.local_decoder = RoFormerEncoder(self.local_config)
        self.global_encoder = RoFormerEncoder(self.global_config)
        
        # Global SOS token
        self.global_sos = nn.Parameter(torch.randn(hidden_size))
        
        # Classification head
        self.classifier = nn.Linear(hidden_size, 1)  # Binary classification
        self.dropout = nn.Dropout(0.1)
        
        # Future mask buffer for autoregressive attention
        self.register_buffer("future_mask", None)
        
    def buffered_future_mask(self, tensor):
        """Create causal mask for autoregressive attention"""
        dim = tensor.size(-2)
        if self.future_mask is None or self.future_mask.size(0) < dim:
            self.future_mask = torch.triu(torch.ones(dim, dim, dtype=torch.bool), diagonal=1).to(tensor.device)
        return self.future_mask[:dim, :dim]
    
    def quantize_duration(self, raw_duration):
        """
        Quantize raw duration values to duration template indices.
        Same logic as preprocessing pipeline.
        """
        # Convert numpy array to torch tensor for compatibility
        duration_templates = torch.tensor(DURATION_TEMPLATES, device=raw_duration.device, dtype=raw_duration.dtype)
        
        # Create boundaries for quantization
        boundaries = (duration_templates[1:] + duration_templates[:-1]) / 2
        
        # Find the closest template index for each duration
        # Use searchsorted to find the insertion point, then adjust
        indices = torch.searchsorted(boundaries, raw_duration.float())
        
        # Clamp to valid range [0, len(DURATION_TEMPLATES)-1]
        indices = torch.clamp(indices, 0, len(DURATION_TEMPLATES) - 1)
        
        return indices.long()

    def preprocess(self, x, pitch_shift=None):
        """
        Preprocess polyphonic data exactly like main model.
        Converts raw duration values to duration indices using templates.
        
        Input: [batch_size, seq_length, 12] polyphonic data
        Output: [batch_size, seq_length, 8] processed tokens
        """
        batch_size, seq_length, subseq_length = x.shape
        
        # Reshape to [batch, seq, 4, 3] for (program, pitch, raw_duration)
        x = x.long().view(batch_size, seq_length, subseq_length // 3, 3)
        
        # Create output tensor [batch, seq, 4, 2]
        x_processed = torch.zeros(
            batch_size, seq_length, subseq_length // 3, 2,
            dtype=torch.long, device=x.device
        )
        
        # Apply same preprocessing as main model
        pad_indices = x[:, :, :, 1] == 255  # pitch is 255 -> pad
        eos_indices = x[:, :, :, 0] == 254  # program is 254 -> eos
        is_not_drum = x[:, :, :, 0] != 127
        
        x_processed[:, :, :, 0] = 0  # program unchanged
        
        # CRITICAL: Quantize raw durations to template indices
        raw_durations = x[:, :, :, 2]
        duration_indices = self.quantize_duration(raw_durations)
        
        # Formula: pitch + duration_index * 128 + 2 + pitch_shift * is_not_drum
        if pitch_shift is None:
            pitch_shift = torch.zeros(batch_size, device=x.device)
        
        x_processed[:, :, :, 1] = (
            x[:, :, :, 1] +  # pitch (0-127)
            duration_indices * 128 +  # duration_index (0-23) * 128
            2 +  # offset
            pitch_shift.unsqueeze(1).unsqueeze(2) * is_not_drum  # pitch shift
        )
        
        # Apply special tokens
        x_processed[pad_indices] = self.PAD_TOKEN
        x_processed[:, :, :, 0][eos_indices] = self.EOS_TOKEN
        
        # Defensive clamping - should not be needed but adds safety
        x_processed = torch.clamp(x_processed, 0, self.vocab_size - 1)
        
        # Flatten to [batch, seq, 8]
        return x_processed.view(batch_size, seq_length, subseq_length // 3 * 2)
    
    def local_encode(self, x, token_type_ids):
        """
        Local encoding step (same as main model).
        Process each frame independently.
        """
        batch_size, seq_len, subseq_len = x.shape
        x = x.view(batch_size * seq_len, subseq_len)
        token_type_ids = token_type_ids.view(batch_size * seq_len, subseq_len + 1)
        
        # Prepend SOS token
        x = torch.cat([
            torch.full((x.shape[0], 1), self.SOS_TOKEN, dtype=torch.long, device=x.device),
            x
        ], dim=-1)
        
        # Create attention mask and embeddings
        mask = x != self.PAD_TOKEN
        word_emb = self.local_embedding(x)
        type_emb = self.token_type_embeddings(token_type_ids)
        type_emb = type_emb.view(batch_size * seq_len, word_emb.shape[1], -1)
        
        # Combine embeddings and encode
        emb = word_emb + type_emb
        h = self.local_encoder(emb, encoder_attention_mask=mask)[0]
        
        return h[:, 0], emb[:, :-1]  # Return SOS representation and embeddings
    
    def local_decode(self, h, emb):
        """Local decoding step (simplified for classification)"""
        batch_size, subseq_len, _ = emb.shape
        h = h.view(batch_size, 1, -1)
        emb = torch.cat([h, emb[:, 1:]], dim=1)
        
        # Apply autoregressive attention
        h = self.local_decoder(emb, attention_mask=self.buffered_future_mask(emb))[0]
        return h
    
    def forward(self, x, pitch_shift=None):
        """
        Forward pass through hierarchical architecture.
        
        Input: 
            x: [batch_size, seq_len, 12] interleaved melody-accompaniment
            pitch_shift: [batch_size] pitch shift values for each sequence
        Output: [batch_size, 1] binary classification logits
        """
        batch_size, seq_len, subseq_len = x.shape
        
        # Ensure even number of frames (interleaved)
        assert seq_len % 2 == 0, "Expected even number of frames (interleaved mel-acc)"
        
        # Preprocess polyphonic data with pitch shift
        x = self.preprocess(x, pitch_shift)  # [batch, seq, 8]
        
        # Create token type embeddings (0=melody, 1=accompaniment)
        idx = torch.arange(seq_len, device=x.device)
        frame_type = (idx % 2 == 0).long()  # even=acc(1), odd=mel(0) -> need to flip
        frame_type = 1 - frame_type  # Now: even=acc(1), odd=mel(0)
        
        token_type_ids = frame_type.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, x.shape[-1])
        sos_type = frame_type.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, 1)
        token_type_ids = torch.cat([sos_type, token_type_ids], dim=-1)
        
        # Local encoding
        h, emb = self.local_encode(x, token_type_ids)
        h = h.view(batch_size, seq_len, -1)
        
        # Global encoding preparation
        sos = self.global_sos.view(1, 1, -1).repeat(batch_size, 1, 1)
        h = torch.cat([sos, h[:, :-1]], dim=1)
        
        # Global encoding with causal attention
        h = self.global_encoder(h, attention_mask=self.buffered_future_mask(h))[0]
        
        # Global pooling - take final token representation
        sequence_representation = h[:, -1]  # [batch_size, hidden_size]
        
        # Classification
        sequence_representation = self.dropout(sequence_representation)
        logits = self.classifier(sequence_representation)  # [batch_size, 1]
        
        return logits.squeeze(-1)  # [batch_size]


def create_interleaved_sequence(melody, accompaniment):
    """
    Create interleaved sequence: [acc_0, mel_0, acc_1, mel_1, ...]
    
    Args:
        melody: [seq_len, 12]
        accompaniment: [seq_len, 12] 
        
    Returns:
        interleaved: [2*seq_len, 12]
    """
    seq_len = min(melody.shape[0], accompaniment.shape[0])
    
    # Handle edge case of empty sequences
    if seq_len == 0:
        return torch.zeros(0, 12, dtype=melody.dtype, device=melody.device)
    
    melody = melody[:seq_len]
    accompaniment = accompaniment[:seq_len]
    
    # Interleave: acc first, then mel for each time step
    interleaved = torch.zeros(2 * seq_len, 12, dtype=melody.dtype, device=melody.device)
    interleaved[0::2] = accompaniment  # Even indices: accompaniment  
    interleaved[1::2] = melody         # Odd indices: melody
    
    return interleaved