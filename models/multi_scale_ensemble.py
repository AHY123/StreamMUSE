import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
from schema.model_schema import MultiScaleEnsembleSchema
from schema.model_io_schema import RewardOutput
from .contrastive_reward_model import ContrastiveRewardModel
from .discriminative_reward_model import DiscriminativeRewardModel
import time
import numpy as np


class MultiScaleEnsemble:
    """
    Multi-scale ensemble system for reward models.
    Manages multiple reward models at different temporal scales with sliding window evaluation.
    """
    
    def __init__(self, ensemble_schema: MultiScaleEnsembleSchema):
        self.ensemble_schema = ensemble_schema
        self.scales = ensemble_schema.scales
        self.overlap_ratio = ensemble_schema.overlap_ratio
        self.aggregation_strategy = ensemble_schema.aggregation_strategy
        self.weights = ensemble_schema.weights or {}
        
        # Dictionary to store models by scale
        self.models: Dict[int, Union[ContrastiveRewardModel, DiscriminativeRewardModel]] = {}
        
    def add_model(self, scale: int, model: Union[ContrastiveRewardModel, DiscriminativeRewardModel]):
        """
        Add a reward model for a specific scale.
        
        Args:
            scale: Frame length this model was trained on
            model: Trained reward model
        """
        if scale not in self.scales:
            raise ValueError(f"Scale {scale} not in configured scales {self.scales}")
        
        self.models[scale] = model
        
    def extract_sliding_windows(self, sequence: torch.Tensor, window_size: int) -> List[torch.Tensor]:
        """
        Extract sliding windows from sequence with specified overlap.
        
        Args:
            sequence: [seq_len] input sequence
            window_size: Size of each window
            
        Returns:
            List of window tensors
        """
        seq_len = len(sequence)
        
        if seq_len <= window_size:
            # If sequence is shorter than window, return the full sequence
            return [sequence]
        
        # Calculate step size based on overlap ratio
        step_size = int(window_size * (1 - self.overlap_ratio))
        if step_size == 0:
            step_size = 1  # Ensure we make progress
        
        windows = []
        start = 0
        
        while start + window_size <= seq_len:
            window = sequence[start:start + window_size]
            windows.append(window)
            start += step_size
        
        # Add final window if there's remaining sequence
        if start < seq_len:
            final_window = sequence[-window_size:]  # Take last window_size tokens
            windows.append(final_window)
            
        return windows
    
    def compute_scale_rewards(self, melody_sequence: torch.Tensor, accompaniment_sequence: torch.Tensor, scale: int) -> List[float]:
        """
        Compute rewards for all windows at a specific scale.
        
        Args:
            melody_sequence: [seq_len] melody tokens
            accompaniment_sequence: [seq_len] accompaniment tokens  
            scale: Frame length scale to evaluate
            
        Returns:
            List of reward scores for each window
        """
        if scale not in self.models:
            raise ValueError(f"No model available for scale {scale}")
        
        model = self.models[scale]
        
        # Extract sliding windows for both sequences
        melody_windows = self.extract_sliding_windows(melody_sequence, scale)
        accompaniment_windows = self.extract_sliding_windows(accompaniment_sequence, scale)
        
        # Ensure same number of windows
        min_windows = min(len(melody_windows), len(accompaniment_windows))
        melody_windows = melody_windows[:min_windows]
        accompaniment_windows = accompaniment_windows[:min_windows]
        
        # Compute reward for each window pair
        rewards = []
        for mel_window, acc_window in zip(melody_windows, accompaniment_windows):
            reward = model.compute_reward(mel_window, acc_window)
            rewards.append(reward)
            
        return rewards
    
    def aggregate_rewards(self, rewards_by_scale: Dict[int, List[float]]) -> float:
        """
        Aggregate rewards from different scales using the configured strategy.
        
        Args:
            rewards_by_scale: Dictionary mapping scale to list of reward scores
            
        Returns:
            float: Aggregated reward score
        """
        if self.aggregation_strategy == "sum":
            total_reward = 0.0
            for scale_rewards in rewards_by_scale.values():
                total_reward += sum(scale_rewards)
            return total_reward
            
        elif self.aggregation_strategy == "mean":
            all_rewards = []
            for scale_rewards in rewards_by_scale.values():
                all_rewards.extend(scale_rewards)
            return np.mean(all_rewards) if all_rewards else 0.0
            
        elif self.aggregation_strategy == "weighted":
            weighted_sum = 0.0
            total_weight = 0.0
            
            for scale, scale_rewards in rewards_by_scale.items():
                weight = self.weights.get(scale, 1.0)
                scale_mean = np.mean(scale_rewards) if scale_rewards else 0.0
                weighted_sum += weight * scale_mean * len(scale_rewards)
                total_weight += weight * len(scale_rewards)
                
            return weighted_sum / total_weight if total_weight > 0 else 0.0
            
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.aggregation_strategy}")
    
    def compute_ensemble_reward(self, melody_sequence: torch.Tensor, accompaniment_sequence: torch.Tensor) -> RewardOutput:
        """
        Compute ensemble reward across all scales.
        
        Args:
            melody_sequence: [seq_len] melody tokens
            accompaniment_sequence: [seq_len] accompaniment tokens
            
        Returns:
            RewardOutput: Aggregated reward with metadata
        """
        start_time = time.time()
        
        if not self.models:
            raise ValueError("No models loaded in ensemble")
        
        # Compute rewards for each scale
        rewards_by_scale = {}
        for scale in self.scales:
            if scale in self.models:
                scale_rewards = self.compute_scale_rewards(melody_sequence, accompaniment_sequence, scale)
                rewards_by_scale[scale] = scale_rewards
        
        # Aggregate across scales
        final_reward = self.aggregate_rewards(rewards_by_scale)
        
        computation_time = time.time() - start_time
        
        # Calculate total windows evaluated
        total_windows = sum(len(rewards) for rewards in rewards_by_scale.values())
        
        return RewardOutput(
            reward_score=final_reward,
            confidence=self._compute_confidence(rewards_by_scale),
            model_type=f"Multi-Scale-Ensemble-{self.aggregation_strategy}",
            sequence_length=len(melody_sequence),
            computation_time=computation_time
        )
    
    def _compute_confidence(self, rewards_by_scale: Dict[int, List[float]]) -> float:
        """
        Compute confidence measure based on reward consistency across scales.
        
        Args:
            rewards_by_scale: Dictionary mapping scale to reward scores
            
        Returns:
            float: Confidence measure (higher = more consistent)
        """
        if len(rewards_by_scale) < 2:
            return 1.0  # Single scale, assume full confidence
        
        # Compute mean reward for each scale
        scale_means = {}
        for scale, rewards in rewards_by_scale.items():
            if rewards:
                scale_means[scale] = np.mean(rewards)
        
        if len(scale_means) < 2:
            return 1.0
        
        # Compute variance across scale means (lower variance = higher confidence)
        means_array = np.array(list(scale_means.values()))
        variance = np.var(means_array)
        
        # Convert variance to confidence (inverse relationship)
        # Use exponential decay: confidence = exp(-variance)
        confidence = np.exp(-variance)
        
        return float(confidence)
    
    def get_scale_breakdown(self, melody_sequence: torch.Tensor, accompaniment_sequence: torch.Tensor) -> Dict[int, Dict[str, float]]:
        """
        Get detailed breakdown of rewards by scale for analysis.
        
        Args:
            melody_sequence: [seq_len] melody tokens
            accompaniment_sequence: [seq_len] accompaniment tokens
            
        Returns:
            Dict mapping scale to reward statistics
        """
        breakdown = {}
        
        for scale in self.scales:
            if scale in self.models:
                scale_rewards = self.compute_scale_rewards(melody_sequence, accompaniment_sequence, scale)
                
                if scale_rewards:
                    breakdown[scale] = {
                        'mean_reward': float(np.mean(scale_rewards)),
                        'std_reward': float(np.std(scale_rewards)),
                        'min_reward': float(np.min(scale_rewards)),
                        'max_reward': float(np.max(scale_rewards)),
                        'num_windows': len(scale_rewards),
                        'rewards': scale_rewards
                    }
                else:
                    breakdown[scale] = {
                        'mean_reward': 0.0,
                        'std_reward': 0.0,
                        'min_reward': 0.0,
                        'max_reward': 0.0,
                        'num_windows': 0,
                        'rewards': []
                    }
        
        return breakdown