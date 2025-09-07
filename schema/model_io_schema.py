from pydantic import BaseModel, Field
from typing import Optional, Any, Union, Iterator
import torch
import numpy as np
from collections.abc import Mapping
from dataclasses import dataclass


class BaseModelInputData(BaseModel):
    """
    Schema for model input.
    """

    mel_data: Optional[Union[torch.Tensor, np.ndarray]] = Field(..., description="Melody data in REMI format.  to the tensor conversion.")
    acc_data: Optional[Union[torch.Tensor, np.ndarray]] = Field(..., description="Accompaniment data in REMI format. to the tensor conversion.")

    model_config = {"arbitrary_types_allowed": True}


class BaseModelOutputData(Mapping, BaseModel):
    """
    Schema for model output.
    """

    logits: Optional[torch.Tensor] = Field(None, description="Logits from the model output.")
    loss: Optional[torch.Tensor] = Field(None, description="Loss value computed from the model output.")
    predicted_ids: Optional[torch.Tensor] = Field(None, description="Predicted IDs from the model output.")
    metadata: Optional[Any] = Field(None, description="Optional metadata associated with the model output.")

    model_config = {"arbitrary_types_allowed": True}

    def __getitem__(self, key: str) -> Any:
        if key in self.model_fields.keys():
            return getattr(self, key)
        raise KeyError(f"Key '{key}' not found in model fields.")

    def __len__(self) -> int:
        return len(self.model_fields)

    def __iter__(self) -> Iterator[str]:
        yield from self.model_fields.keys()


class M2AModelInputData(BaseModelInputData):
    """
    Schema for M2A Transformer model input.
    """

    pitch_shift: Optional[torch.Tensor] = Field(None, description="Pitch shift values for the melody and accompaniment data.")

class NewPtM2AModelInputData(M2AModelInputData):
    """
    Schema for M2A Transformer model input.
    """



class M2AModelOutputData(BaseModelOutputData):
    """
    Schema for M2A Transformer model output.
    """

class NewPtM2AModelOutputData(BaseModelOutputData):
    """
    Schema for M2A Transformer model output.
    """



# Reward Model Data Structures

@dataclass
class ContrastiveBatch:
    """Batch structure for contrastive reward model training."""
    melody_sequences: torch.Tensor      # [B, seq_len]
    accompaniment_sequences: torch.Tensor  # [B, seq_len]  
    pair_labels: torch.Tensor           # [B] - 1 for positive, 0 for negative
    sequence_lengths: torch.Tensor      # [B]


@dataclass 
class DiscriminativeBatch:
    """Batch structure for discriminative reward model training."""
    interleaved_sequences: torch.Tensor  # [B, seq_len]
    labels: torch.Tensor                # [B] - 1 for real, 0 for fake
    attention_masks: torch.Tensor       # [B, seq_len]
    sequence_lengths: torch.Tensor      # [B]


@dataclass
class RewardOutput:
    """Standardized output structure for reward computation."""
    reward_score: float
    confidence: Optional[float] = None
    model_type: str = ""
    sequence_length: int = 0
    computation_time: float = 0.0


# Batch collation functions
def collate_contrastive_batch(batch_items) -> ContrastiveBatch:
    """Collate function for contrastive training batches."""
    melody_seqs = torch.stack([item['melody'] for item in batch_items]).long()
    acc_seqs = torch.stack([item['accompaniment'] for item in batch_items]).long()
    labels = torch.tensor([item['label'] for item in batch_items]).long()
    seq_lens = torch.tensor([item['length'] for item in batch_items]).long()
    
    return ContrastiveBatch(
        melody_sequences=melody_seqs,
        accompaniment_sequences=acc_seqs,
        pair_labels=labels,
        sequence_lengths=seq_lens
    )


def collate_discriminative_batch(batch_items) -> DiscriminativeBatch:
    """Collate function for discriminative training batches."""
    interleaved_seqs = torch.stack([item['sequence'] for item in batch_items]).long()
    labels = torch.tensor([item['label'] for item in batch_items]).long()
    attention_masks = torch.stack([item['mask'] for item in batch_items]).float()
    seq_lens = torch.tensor([item['length'] for item in batch_items]).long()
    
    return DiscriminativeBatch(
        interleaved_sequences=interleaved_seqs,
        labels=labels,
        attention_masks=attention_masks,
        sequence_lengths=seq_lens
    )


ModelInputData = Union[M2AModelInputData, NewPtM2AModelInputData]
ModelOutputData = Union[M2AModelOutputData, NewPtM2AModelInputData]
