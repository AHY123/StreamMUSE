import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union
import random
from pathlib import Path
import json
from models.contrastive_reward_model import ContrastiveRewardModel
from models.discriminative_reward_model import DiscriminativeRewardModel
from models.multi_scale_ensemble import MultiScaleEnsemble


class PerturbationValidator:
    """
    Validation framework for reward models using progressive accompaniment corruption.
    Tests if models can distinguish between high and low quality melody-accompaniment pairs.
    """
    
    def __init__(self):
        self.perturbation_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.results_history = []
    
    def corrupt_accompaniment(self, accompaniment: torch.Tensor, corruption_ratio: float) -> torch.Tensor:
        """
        Progressively corrupt accompaniment by replacing tokens with random tokens.
        
        Args:
            accompaniment: [seq_len] original accompaniment sequence
            corruption_ratio: Fraction of tokens to corrupt (0.0 to 1.0)
            
        Returns:
            torch.Tensor: Corrupted accompaniment sequence
        """
        if corruption_ratio <= 0:
            return accompaniment.clone()
        
        corrupted = accompaniment.clone()
        seq_len = len(corrupted)
        
        # Don't corrupt padding tokens (3204)
        non_pad_mask = corrupted != 3204
        non_pad_indices = torch.where(non_pad_mask)[0]
        
        if len(non_pad_indices) == 0:
            return corrupted
        
        # Number of tokens to corrupt
        num_to_corrupt = int(len(non_pad_indices) * corruption_ratio)
        
        if num_to_corrupt > 0:
            # Randomly select indices to corrupt from non-padding tokens
            corrupt_indices = random.sample(non_pad_indices.tolist(), min(num_to_corrupt, len(non_pad_indices)))
            
            # Replace with random tokens from valid vocabulary (excluding special tokens)
            # Use tokens from range 3 to 3201 (avoiding SOS=3202, EOS=3203, PAD=3204)
            for idx in corrupt_indices:
                corrupted[idx] = random.randint(3, 3201)
        
        return corrupted
    
    def validate_single_model(
        self, 
        model: Union[ContrastiveRewardModel, DiscriminativeRewardModel],
        test_pairs: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, List[float]]:
        """
        Run perturbation validation on a single reward model.
        
        Args:
            model: Reward model to validate
            test_pairs: List of (melody, accompaniment) pairs from test set
            
        Returns:
            Dict with perturbation levels and corresponding mean rewards
        """
        model.eval()
        
        results = {
            'corruption_levels': [0.0] + self.perturbation_levels,
            'mean_rewards': [],
            'std_rewards': [],
            'all_rewards': []  # Store all individual rewards for analysis
        }
        
        # Test at each corruption level
        for corruption_ratio in results['corruption_levels']:
            level_rewards = []
            
            for melody, accompaniment in test_pairs:
                # Create corrupted version
                corrupted_acc = self.corrupt_accompaniment(accompaniment, corruption_ratio)
                
                # Compute reward
                reward = model.compute_reward(melody, corrupted_acc)
                level_rewards.append(reward)
            
            # Store statistics
            results['mean_rewards'].append(np.mean(level_rewards))
            results['std_rewards'].append(np.std(level_rewards))
            results['all_rewards'].append(level_rewards)
            
        return results
    
    def validate_ensemble(
        self,
        ensemble: MultiScaleEnsemble,
        test_pairs: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, List[float]]:
        """
        Run perturbation validation on multi-scale ensemble.
        
        Args:
            ensemble: Multi-scale ensemble to validate
            test_pairs: List of (melody, accompaniment) pairs from test set
            
        Returns:
            Dict with perturbation results including scale breakdown
        """
        results = {
            'corruption_levels': [0.0] + self.perturbation_levels,
            'mean_rewards': [],
            'std_rewards': [],
            'all_rewards': [],
            'scale_breakdown': {scale: {'mean_rewards': [], 'std_rewards': []} 
                              for scale in ensemble.scales}
        }
        
        # Test at each corruption level
        for corruption_ratio in results['corruption_levels']:
            level_rewards = []
            scale_rewards = {scale: [] for scale in ensemble.scales}
            
            for melody, accompaniment in test_pairs:
                # Create corrupted version
                corrupted_acc = self.corrupt_accompaniment(accompaniment, corruption_ratio)
                
                # Get ensemble reward and scale breakdown
                ensemble_result = ensemble.compute_ensemble_reward(melody, corrupted_acc)
                level_rewards.append(ensemble_result.reward_score)
                
                # Get breakdown by scale for analysis
                breakdown = ensemble.get_scale_breakdown(melody, corrupted_acc)
                for scale, stats in breakdown.items():
                    if stats['num_windows'] > 0:
                        scale_rewards[scale].append(stats['mean_reward'])
            
            # Store ensemble results
            results['mean_rewards'].append(np.mean(level_rewards))
            results['std_rewards'].append(np.std(level_rewards))
            results['all_rewards'].append(level_rewards)
            
            # Store scale-specific results
            for scale in ensemble.scales:
                if scale in scale_rewards and scale_rewards[scale]:
                    results['scale_breakdown'][scale]['mean_rewards'].append(np.mean(scale_rewards[scale]))
                    results['scale_breakdown'][scale]['std_rewards'].append(np.std(scale_rewards[scale]))
                else:
                    results['scale_breakdown'][scale]['mean_rewards'].append(0.0)
                    results['scale_breakdown'][scale]['std_rewards'].append(0.0)
                    
        return results
    
    def analyze_trend(self, corruption_levels: List[float], mean_rewards: List[float]) -> Dict[str, float]:
        """
        Analyze reward degradation trend.
        
        Args:
            corruption_levels: List of corruption ratios
            mean_rewards: Corresponding mean reward values
            
        Returns:
            Dict with trend analysis metrics
        """
        if len(corruption_levels) != len(mean_rewards) or len(corruption_levels) < 2:
            return {'slope': 0.0, 'correlation': 0.0, 'degradation_detected': False}
        
        # Compute linear trend (slope)
        x = np.array(corruption_levels)
        y = np.array(mean_rewards)
        
        # Linear regression
        slope, intercept = np.polyfit(x, y, 1)
        
        # Correlation coefficient
        correlation = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0.0
        
        # Check if clear degradation trend exists
        # For a good reward model, we expect:
        # 1. Negative slope (rewards decrease with corruption)
        # 2. Strong negative correlation
        degradation_detected = slope < -0.1 and correlation < -0.5
        
        return {
            'slope': float(slope),
            'intercept': float(intercept),
            'correlation': float(correlation),
            'degradation_detected': degradation_detected,
            'initial_reward': float(y[0]) if len(y) > 0 else 0.0,
            'final_reward': float(y[-1]) if len(y) > 0 else 0.0,
            'total_degradation': float(y[0] - y[-1]) if len(y) > 0 else 0.0
        }
    
    def plot_perturbation_results(
        self, 
        results: Dict[str, List[float]], 
        model_name: str,
        save_path: str = None
    ) -> plt.Figure:
        """
        Create plot showing reward degradation with perturbation.
        
        Args:
            results: Validation results from validate_single_model or validate_ensemble
            model_name: Name for plot title
            save_path: Optional path to save plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        corruption_levels = results['corruption_levels']
        mean_rewards = results['mean_rewards']
        std_rewards = results['std_rewards']
        
        # Plot mean with error bars
        ax.errorbar(corruption_levels, mean_rewards, yerr=std_rewards, 
                   marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
        
        # Add trend analysis
        trend = self.analyze_trend(corruption_levels, mean_rewards)
        
        ax.set_xlabel('Corruption Ratio', fontsize=12)
        ax.set_ylabel('Mean Reward Score', fontsize=12)
        ax.set_title(f'{model_name} - Perturbation Validation\n'
                    f'Slope: {trend["slope"]:.3f}, Correlation: {trend["correlation"]:.3f}', 
                    fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 0.95)
        
        # Add degradation status text
        status_text = "✓ Clear Degradation" if trend['degradation_detected'] else "⚠ Weak Degradation"
        ax.text(0.02, 0.98, status_text, transform=ax.transAxes, 
               bbox=dict(boxstyle="round,pad=0.3", 
                        facecolor="green" if trend['degradation_detected'] else "orange", 
                        alpha=0.7),
               verticalalignment='top', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def save_results(self, results: Dict, output_path: str):
        """Save validation results to JSON file."""
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        results_json = convert_numpy(results)
        
        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=2)
    
    def run_full_validation(
        self,
        model: Union[ContrastiveRewardModel, DiscriminativeRewardModel, MultiScaleEnsemble],
        test_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        model_name: str,
        output_dir: str = "validation_results"
    ) -> Dict[str, any]:
        """
        Run complete perturbation validation with analysis and visualization.
        
        Args:
            model: Model to validate
            test_pairs: Test data pairs
            model_name: Name for output files
            output_dir: Directory to save results
            
        Returns:
            Complete validation results with trend analysis
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Run validation
        if isinstance(model, MultiScaleEnsemble):
            results = self.validate_ensemble(model, test_pairs)
        else:
            results = self.validate_single_model(model, test_pairs)
        
        # Analyze trend
        trend_analysis = self.analyze_trend(results['corruption_levels'], results['mean_rewards'])
        results['trend_analysis'] = trend_analysis
        
        # Create and save plot
        fig = self.plot_perturbation_results(results, model_name)
        plot_path = output_path / f"{model_name}_perturbation_validation.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Save detailed results
        results_path = output_path / f"{model_name}_validation_results.json"
        self.save_results(results, str(results_path))
        
        # Store in history
        self.results_history.append({
            'model_name': model_name,
            'results': results,
            'timestamp': str(Path.cwd())  # Placeholder for timestamp
        })
        
        return results