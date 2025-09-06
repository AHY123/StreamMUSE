import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import List, Dict, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from models.contrastive_reward_model import ContrastiveRewardModel
from models.discriminative_reward_model import DiscriminativeRewardModel
from models.multi_scale_ensemble import MultiScaleEnsemble
import json
from pathlib import Path


class RewardModelEvaluator:
    """
    Comprehensive evaluation system for reward models.
    Provides metrics for both contrastive and discriminative models.
    """
    
    def __init__(self):
        self.results = {}
    
    def evaluate_contrastive_model(
        self,
        model: ContrastiveRewardModel,
        test_pairs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]],  # (melody, acc_pos, acc_neg, correct_idx)
        num_candidates: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate contrastive model on accompaniment matching task.
        
        Args:
            model: Contrastive reward model
            test_pairs: List of (melody, positive_acc, negative_acc, correct_index) tuples
            num_candidates: Number of candidate accompaniments to test
            
        Returns:
            Dict with evaluation metrics
        """
        model.eval()
        
        correct_predictions = 0
        total_predictions = 0
        all_similarities = []
        all_labels = []
        
        with torch.no_grad():
            for melody, acc_pos, acc_neg, correct_idx in test_pairs:
                # Create candidate list (positive + negatives)
                candidates = [acc_pos, acc_neg]
                
                # Add more random negatives if needed
                while len(candidates) < num_candidates:
                    # Get random accompaniment from test set
                    rand_idx = np.random.randint(0, len(test_pairs))
                    rand_acc = test_pairs[rand_idx][1]  # Get positive acc from random pair
                    candidates.append(rand_acc)
                
                # Compute similarities for all candidates
                similarities = []
                for candidate in candidates:
                    similarity = model.compute_reward(melody, candidate)
                    similarities.append(similarity)
                
                # Find highest similarity
                predicted_idx = np.argmax(similarities)
                
                # Check if correct (positive accompaniment should have highest similarity)
                if predicted_idx == 0:  # First candidate is always the positive one
                    correct_predictions += 1
                
                total_predictions += 1
                
                # Store for additional metrics
                all_similarities.extend(similarities)
                # Label: 1 for positive, 0 for negatives
                labels = [1] + [0] * (len(candidates) - 1)
                all_labels.extend(labels)
        
        # Calculate metrics
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        # Convert similarities to binary predictions (above median = positive)
        similarity_threshold = np.median(all_similarities)
        binary_predictions = [1 if s >= similarity_threshold else 0 for s in all_similarities]
        
        precision = precision_score(all_labels, binary_predictions, zero_division=0)
        recall = recall_score(all_labels, binary_predictions, zero_division=0)
        f1 = f1_score(all_labels, binary_predictions, zero_division=0)
        
        # AUC if possible
        try:
            auc = roc_auc_score(all_labels, all_similarities)
        except ValueError:
            auc = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'mean_similarity': np.mean(all_similarities),
            'std_similarity': np.std(all_similarities),
            'num_test_samples': total_predictions
        }
    
    def evaluate_discriminative_model(
        self,
        model: DiscriminativeRewardModel,
        test_pairs: List[Tuple[torch.Tensor, torch.Tensor, int]]  # (melody, accompaniment, label)
    ) -> Dict[str, float]:
        """
        Evaluate discriminative model on binary classification task.
        
        Args:
            model: Discriminative reward model
            test_pairs: List of (melody, accompaniment, true_label) tuples
            
        Returns:
            Dict with binary classification metrics
        """
        model.eval()
        
        predictions = []
        true_labels = []
        probabilities = []
        
        with torch.no_grad():
            for melody, accompaniment, true_label in test_pairs:
                # Get model prediction (P(real))
                prob_real = model.compute_reward(melody, accompaniment)
                
                # Convert to binary prediction
                pred_label = 1 if prob_real >= 0.5 else 0
                
                predictions.append(pred_label)
                true_labels.append(true_label)
                probabilities.append(prob_real)
        
        # Calculate standard binary classification metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        
        # AUC
        try:
            auc = roc_auc_score(true_labels, probabilities)
        except ValueError:
            auc = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'mean_probability': np.mean(probabilities),
            'std_probability': np.std(probabilities),
            'num_test_samples': len(test_pairs)
        }
    
    def analyze_reward_distribution(
        self,
        model: Union[ContrastiveRewardModel, DiscriminativeRewardModel],
        test_pairs: List[Tuple[torch.Tensor, torch.Tensor, int]]
    ) -> Dict[str, any]:
        """
        Analyze the distribution of reward scores.
        
        Args:
            model: Reward model to analyze
            test_pairs: Test data pairs
            
        Returns:
            Dict with distribution statistics
        """
        model.eval()
        
        positive_rewards = []
        negative_rewards = []
        
        with torch.no_grad():
            for melody, accompaniment, label in test_pairs:
                reward = model.compute_reward(melody, accompaniment)
                
                if label == 1:
                    positive_rewards.append(reward)
                else:
                    negative_rewards.append(reward)
        
        # Compute statistics for each class
        pos_stats = {
            'mean': np.mean(positive_rewards) if positive_rewards else 0.0,
            'std': np.std(positive_rewards) if positive_rewards else 0.0,
            'min': np.min(positive_rewards) if positive_rewards else 0.0,
            'max': np.max(positive_rewards) if positive_rewards else 0.0,
            'count': len(positive_rewards)
        }
        
        neg_stats = {
            'mean': np.mean(negative_rewards) if negative_rewards else 0.0,
            'std': np.std(negative_rewards) if negative_rewards else 0.0,
            'min': np.min(negative_rewards) if negative_rewards else 0.0,
            'max': np.max(negative_rewards) if negative_rewards else 0.0,
            'count': len(negative_rewards)
        }
        
        # Separation metrics
        all_rewards = positive_rewards + negative_rewards
        separation_score = abs(pos_stats['mean'] - neg_stats['mean']) if positive_rewards and negative_rewards else 0.0
        
        return {
            'positive_class': pos_stats,
            'negative_class': neg_stats,
            'separation_score': separation_score,
            'total_samples': len(all_rewards),
            'class_balance': len(positive_rewards) / len(all_rewards) if all_rewards else 0.0
        }
    
    def plot_reward_distributions(
        self,
        model: Union[ContrastiveRewardModel, DiscriminativeRewardModel],
        test_pairs: List[Tuple[torch.Tensor, torch.Tensor, int]],
        model_name: str,
        save_path: str = None
    ) -> plt.Figure:
        """
        Create distribution plots for reward scores.
        
        Args:
            model: Reward model
            test_pairs: Test data pairs
            model_name: Name for plot title
            save_path: Optional save path
            
        Returns:
            matplotlib Figure object
        """
        # Get rewards by class
        positive_rewards = []
        negative_rewards = []
        
        model.eval()
        with torch.no_grad():
            for melody, accompaniment, label in test_pairs:
                reward = model.compute_reward(melody, accompaniment)
                if label == 1:
                    positive_rewards.append(reward)
                else:
                    negative_rewards.append(reward)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram comparison
        if positive_rewards and negative_rewards:
            ax1.hist(positive_rewards, bins=30, alpha=0.7, label='Positive (Real)', color='green')
            ax1.hist(negative_rewards, bins=30, alpha=0.7, label='Negative (Fake)', color='red')
            ax1.set_xlabel('Reward Score')
            ax1.set_ylabel('Frequency')
            ax1.set_title(f'{model_name} - Reward Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Box plot comparison
            data_to_plot = [positive_rewards, negative_rewards]
            labels = ['Positive', 'Negative']
            
            bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
            bp['boxes'][0].set_facecolor('green')
            bp['boxes'][0].set_alpha(0.7)
            bp['boxes'][1].set_facecolor('red') 
            bp['boxes'][1].set_alpha(0.7)
            
            ax2.set_ylabel('Reward Score')
            ax2.set_title(f'{model_name} - Reward Distribution (Box Plot)')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def compute_statistical_significance(
        self,
        positive_rewards: List[float],
        negative_rewards: List[float]
    ) -> Dict[str, float]:
        """
        Compute statistical significance tests between positive and negative rewards.
        
        Args:
            positive_rewards: Rewards for positive samples
            negative_rewards: Rewards for negative samples
            
        Returns:
            Dict with test statistics
        """
        from scipy.stats import ttest_ind, mannwhitneyu
        
        results = {}
        
        if len(positive_rewards) > 1 and len(negative_rewards) > 1:
            # T-test (assumes normal distribution)
            t_stat, t_pvalue = ttest_ind(positive_rewards, negative_rewards)
            results['t_test'] = {'statistic': float(t_stat), 'p_value': float(t_pvalue)}
            
            # Mann-Whitney U test (non-parametric)
            try:
                u_stat, u_pvalue = mannwhitneyu(positive_rewards, negative_rewards, alternative='two-sided')
                results['mann_whitney'] = {'statistic': float(u_stat), 'p_value': float(u_pvalue)}
            except ValueError:
                results['mann_whitney'] = {'statistic': 0.0, 'p_value': 1.0}
        
        return results
    
    def run_comprehensive_evaluation(
        self,
        model: Union[ContrastiveRewardModel, DiscriminativeRewardModel, MultiScaleEnsemble],
        test_data: Dict[str, List],
        model_name: str,
        output_dir: str = "evaluation_results"
    ) -> Dict[str, any]:
        """
        Run complete evaluation suite for a reward model.
        
        Args:
            model: Model to evaluate
            test_data: Test data in appropriate format
            model_name: Name for output files
            output_dir: Directory to save results
            
        Returns:
            Complete evaluation results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results = {'model_name': model_name, 'model_type': type(model).__name__}
        
        # Run appropriate evaluation based on model type
        if isinstance(model, ContrastiveRewardModel):
            if 'contrastive_pairs' in test_data:
                results['task_metrics'] = self.evaluate_contrastive_model(
                    model, test_data['contrastive_pairs']
                )
        elif isinstance(model, DiscriminativeRewardModel):
            if 'binary_pairs' in test_data:
                results['task_metrics'] = self.evaluate_discriminative_model(
                    model, test_data['binary_pairs']
                )
        
        # Distribution analysis (common for both)
        if 'binary_pairs' in test_data:
            results['distribution_analysis'] = self.analyze_reward_distribution(
                model, test_data['binary_pairs']
            )
            
            # Create distribution plots
            fig = self.plot_reward_distributions(
                model, test_data['binary_pairs'], model_name
            )
            plot_path = output_path / f"{model_name}_reward_distributions.png"
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        # Save results
        results_path = output_path / f"{model_name}_evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results