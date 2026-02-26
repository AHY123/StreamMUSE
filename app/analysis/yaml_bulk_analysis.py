#!/usr/bin/env python3
"""
YAML-based StreamMUSE Bulk Analysis Tool

Analyzes multiple experiments based on YAML configuration with custom formulas and formal names.
Creates box plots with fitted lines and shaded confidence bands.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Import the existing bulk analysis class for data loading functionality
sys.path.append(str(Path(__file__).parent))
from bulk_analysis import BulkGenerationLengthAnalyzer

class YAMLBulkAnalysisEngine:
    """YAML-configured bulk analysis engine with formula-based fitting."""
    
    def __init__(self, config_path: str):
        """Initialize with YAML configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.experiments_data = {}
        self.combined_data = None
        
        # Set up output directory
        self.output_dir = Path(self.config['analysis_settings']['output_directory'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {self.config_path}: {e}")
    
    def _apply_anomaly_filter(self, df: pd.DataFrame, experiment_name: str, filter_pct: float) -> pd.DataFrame:
        """Apply anomaly filtering per generation length."""
        if filter_pct <= 0 or 'round_trip_time' not in df.columns:
            return df
        
        if 'generation_length' not in df.columns:
            print(f"⚠️ No generation_length column found for {experiment_name}, skipping anomaly filtering")
            return df
        
        original_count = len(df)
        filtered_data = []
        
        print(f"🔍 Anomaly filtering for {experiment_name} ({filter_pct}% top outliers per generation length):")
        
        for gen_length in sorted(df['generation_length'].unique()):
            gl_data = df[df['generation_length'] == gen_length]
            gl_original_count = len(gl_data)
            
            if gl_original_count < 10:  # Skip filtering if too few samples
                print(f"   GL {gen_length}: {gl_original_count} samples (too few, no filtering)")
                filtered_data.append(gl_data)
                continue
            
            # Calculate upper threshold only (remove top 5% outliers)
            upper_threshold = gl_data['round_trip_time'].quantile(1 - filter_pct / 100)
            
            # Filter data for this generation length - only remove top outliers
            gl_filtered = gl_data[
                gl_data['round_trip_time'] <= upper_threshold
            ].copy()
            
            gl_filtered_count = len(gl_filtered)
            gl_removed_count = gl_original_count - gl_filtered_count
            
            print(f"   GL {gen_length}: {gl_original_count} → {gl_filtered_count} samples "
                  f"(removed {gl_removed_count}, {gl_removed_count/gl_original_count*100:.1f}%)")
            
            filtered_data.append(gl_filtered)
        
        combined_filtered = pd.concat(filtered_data, ignore_index=True)
        total_removed = original_count - len(combined_filtered)
        print(f"   Total for {experiment_name}: {original_count} → {len(combined_filtered)} samples "
              f"(removed {total_removed}, {total_removed/original_count*100:.1f}%)")
        
        return combined_filtered
    
    def load_experiments(self):
        """Load all experiments specified in the YAML configuration."""
        print("📊 YAML-Based StreamMUSE Bulk Analysis")
        print("=" * 50)
        
        anomaly_filter_pct = self.config['analysis_settings'].get('anomaly_filter_percentage', 0)
        if anomaly_filter_pct > 0:
            print(f"🔍 Anomaly filtering enabled: removing {anomaly_filter_pct}% from each tail")
        
        print(f"🔍 Loading {len(self.config['experiments'])} experiments from YAML configuration...")
        
        # Use the existing BulkGenerationLengthAnalyzer for data loading
        experiment_paths = [exp['path'] for exp in self.config['experiments']]
        bulk_engine = BulkGenerationLengthAnalyzer(experiment_paths, self.output_dir, anomaly_filter_pct)
        bulk_engine.load_all_experiments()
        bulk_engine.combine_all_data()
        
        # Extract the loaded data and apply additional processing
        if bulk_engine.combined_data is not None:
            for i, exp_config in enumerate(self.config['experiments']):
                # Use explicit experiment_source if provided, otherwise derive from path
                if 'experiment_source' in exp_config:
                    exp_name = exp_config['experiment_source']
                else:
                    exp_name = Path(exp_config['path']).name
                formal_name = exp_config['formal_name']
                
                # Get the data for this experiment
                exp_data = bulk_engine.combined_data[
                    bulk_engine.combined_data['experiment_source'] == exp_name
                ].copy()
                
                if len(exp_data) > 0:
                    # Store with formal name mapping
                    exp_data['formal_name'] = formal_name
                    exp_data['color'] = exp_config['color']
                    self.experiments_data[formal_name] = {
                        'data': exp_data,
                        'config': exp_config,
                        'original_name': exp_name
                    }
                    print(f"✅ Successfully loaded {formal_name}: {len(exp_data)} samples")
                else:
                    print(f"❌ No data found for {formal_name} (experiment_source: {exp_name})")
        else:
            print("❌ No combined data available from bulk engine")
        
        if self.experiments_data:
            # Combine all data for analysis
            all_data = []
            for formal_name, exp_info in self.experiments_data.items():
                all_data.append(exp_info['data'])
            self.combined_data = pd.concat(all_data, ignore_index=True)
            print(f"✅ Combined {len(self.experiments_data)} experiments: {len(self.combined_data)} total samples")
        else:
            raise ValueError("No experiment data loaded successfully")
    
    def calculate_formula_values(self, gl_values: np.ndarray, formula_config: Dict[str, Any]) -> np.ndarray:
        """Calculate formula values for given generation lengths."""
        a = formula_config.get('a', 0)
        b = formula_config.get('b', 0) 
        c = formula_config.get('c', 0)
        
        if formula_config.get('type') == 'quadratic':
            return a * gl_values**2 + b * gl_values + c
        else:  # linear
            return b * gl_values + c
    
    def plot_experiment_fitting(self):
        """Create the main box plot + line fitting visualization."""
        print("📊 Generating experiment fitting visualization...")
        
        plot_config = self.config['plot_settings']
        
        # Use a more professional style with larger figure for single-column
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(8, 6))  # Single-column optimized size
        
        # Get all unique generation lengths
        all_gl = sorted(self.combined_data['generation_length'].unique())
        
        # Create grouped box plots with clear spacing between generation lengths
        num_experiments = len(self.experiments_data)
        
        # Use a grouped approach - each GL gets a group with clear spacing
        group_width = 1.2  # Total width allocated for each generation length group
        box_width = group_width / (num_experiments + 1)  # Leave space between groups
        group_spacing = 0.4  # Extra space between groups
        
        colors = []
        all_box_data = []
        all_positions = []
        all_colors = []
        
        # Collect all data first
        for i, (formal_name, exp_info) in enumerate(self.experiments_data.items()):
            exp_data = exp_info['data']
            color = exp_info['config']['color']
            colors.append(color)
            
            for j, gl in enumerate(all_gl):
                gl_data = exp_data[exp_data['generation_length'] == gl]
                if len(gl_data) > 0:
                    # Convert to milliseconds for display
                    rtt_ms = gl_data['round_trip_time'] * 1000
                    
                    # Calculate position within the group for this GL
                    group_center = gl * (group_width + group_spacing)
                    box_position = group_center + (i - (num_experiments - 1) / 2) * box_width
                    
                    all_box_data.append(rtt_ms.values)
                    all_positions.append(box_position)
                    all_colors.append(color)
        
        # Create all box plots in one call for consistent styling
        if all_box_data:
            bp = ax.boxplot(all_box_data, positions=all_positions, patch_artist=True, 
                           widths=box_width * 0.8, showfliers=False, 
                           medianprops=dict(color='white', linewidth=2.5),
                           whiskerprops=dict(linewidth=2.5),
                           capprops=dict(linewidth=2.5))
            
            # Color each box with its corresponding experiment color
            for i, (patch, color) in enumerate(zip(bp['boxes'], all_colors)):
                patch.set_facecolor(color)
                patch.set_alpha(0.9)
                patch.set_edgecolor(color)
                patch.set_linewidth(3)
                
            # Color whiskers and caps to match their boxes
            for i, color in enumerate(all_colors):
                bp['whiskers'][i*2].set_color(color)
                bp['whiskers'][i*2+1].set_color(color)
                bp['caps'][i*2].set_color(color)
                bp['caps'][i*2+1].set_color(color)
        
        # Add fitted lines with two-line shading - use the grouped spacing
        gl_values_for_fitting = np.linspace(min(all_gl), max(all_gl), 200)
        gl_positions_for_fitting = [gl * (group_width + group_spacing) for gl in gl_values_for_fitting]
        
        for formal_name, exp_info in self.experiments_data.items():
            formula_config = exp_info['config']['formula']
            color = exp_info['config']['color']
            
            # Calculate main fitted line (100% data)
            fitted_values_main = self.calculate_formula_values(gl_values_for_fitting, formula_config)
            
            # Calculate second line (95% data) from YAML config if available
            if 'formula_95' in exp_info['config']:
                formula_95_config = exp_info['config']['formula_95']
                fitted_values_outline = self.calculate_formula_values(gl_values_for_fitting, formula_95_config)
            else:
                # Fallback if formula_95 not provided
                fitted_values_outline = fitted_values_main * 1.1
            
            # Plot main fitted line with thinner line for better shaded area visibility
            line_label = f"{formal_name}"
            ax.plot(gl_positions_for_fitting, fitted_values_main, color=color, linewidth=plot_config['line_width']*0.7,
                   label=line_label, alpha=0.9, linestyle='-')
            
            # Plot outline/boundary line (95% data)
            ax.plot(gl_positions_for_fitting, fitted_values_outline, color=color, linewidth=plot_config['line_width']*0.6,
                   alpha=0.7, linestyle='--')
            
            # Add shaded area between the two lines
            ax.fill_between(gl_positions_for_fitting, fitted_values_main, fitted_values_outline, color=color, 
                           alpha=plot_config['shaded_area_alpha'], linestyle='None')
        
        # Customize plot appearance with larger fonts for publication
        ax.set_xlabel('Generation Length (Frames)', fontsize=18, fontweight='bold')
        ax.set_ylabel('Round Trip Time (ms)', fontsize=18, fontweight='bold')
        # ax.set_title('RT Model vs Empirical Data', fontsize=20, fontweight='bold', pad=20)
        
        # Set x-axis to show generation length labels at group centers
        group_centers = [gl * (group_width + group_spacing) for gl in all_gl]
        ax.set_xlim(group_centers[0] - group_width, group_centers[-1] + group_width)
        ax.set_xticks(group_centers)
        ax.set_xticklabels(all_gl, fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        
        # Improve grid appearance
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
        ax.set_facecolor('white')
        
        # Create a comprehensive legend with consistent styling and larger font
        handles, labels = ax.get_legend_handles_labels()
        # Keep only the main fit labels (solid lines)
        fit_handles = []
        fit_labels = []
        for handle, label in zip(handles, labels):
            fit_handles.append(handle)
            fit_labels.append(label)
        
        # Add a representative dashed line to explain what it represents
        from matplotlib.lines import Line2D
        dashed_line = Line2D([0], [0], color='gray', linewidth=plot_config['line_width']*0.8, 
                           linestyle='--', alpha=0.7, label='95% Data Fit (Outliers Removed)')
        fit_handles.append(dashed_line)
        fit_labels.append('95% Data Fit (Outliers Removed)')
        
        legend = ax.legend(fit_handles, fit_labels, loc='upper left', fontsize=15, 
                          frameon=True, fancybox=True, shadow=False, framealpha=0.9)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('black')
        
        # Improve y-axis limits for better visualization
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(max(0, y_min * 0.95), y_max * 1.05)
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.plots_dir / "experiment_fitting_analysis.png"
        output_path_pdf = output_path.with_suffix('.pdf')
        plt.savefig(output_path, dpi=plot_config['dpi'], bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Experiment fitting plot saved to: {output_path}")
        
        # Print fitting statistics
        print("\n📈 Formula Fitting Summary:")
        for formal_name, exp_info in self.experiments_data.items():
            formula_config = exp_info['config']['formula']
            exp_data = exp_info['data']
            
            # Calculate R² for the fit
            gl_values = exp_data['generation_length'].values
            actual_rtt = exp_data['round_trip_time'].values * 1000  # Convert to ms
            predicted_rtt = self.calculate_formula_values(gl_values, formula_config)
            
            # Calculate R-squared
            ss_res = np.sum((actual_rtt - predicted_rtt) ** 2)
            ss_tot = np.sum((actual_rtt - np.mean(actual_rtt)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            print(f"  {formal_name}:")
            print(f"    Formula: {formula_config['type']}")
            print(f"    R² = {r_squared:.3f}")
            print(f"    Samples: {len(exp_data)}")
    
    def generate_summary_report(self):
        """Generate a summary report of the YAML-based analysis."""
        report_path = self.output_dir / "yaml_analysis_summary.md"
        
        with open(report_path, 'w') as f:
            f.write("# YAML-Based StreamMUSE Analysis Report\n\n")
            f.write(f"Generated from configuration: `{self.config_path.name}`\n\n")
            
            f.write("## Experiment Configuration\n\n")
            for formal_name, exp_info in self.experiments_data.items():
                config = exp_info['config']
                f.write(f"### {formal_name}\n")
                f.write(f"- **Source**: {config['path']}\n")
                f.write(f"- **Description**: {config['description']}\n")
                f.write(f"- **Formula Type**: {config['formula']['type']}\n")
                if config['formula']['type'] == 'quadratic':
                    f.write(f"- **Formula**: RT = {config['formula']['a']:.2f}×GL² + {config['formula']['b']:.2f}×GL + {config['formula']['c']:.1f}\n")
                else:
                    f.write(f"- **Formula**: RT = {config['formula']['b']:.2f}×GL + {config['formula']['c']:.1f}\n")
                f.write(f"- **Color**: {config['color']}\n")
                f.write(f"- **Samples**: {len(exp_info['data'])}\n\n")
            
            f.write("## Analysis Settings\n\n")
            f.write(f"- **Anomaly Filtering**: {self.config['analysis_settings']['anomaly_filter_percentage']}% per tail\n")
            f.write(f"- **Output Directory**: {self.config['analysis_settings']['output_directory']}\n")
            f.write(f"- **Individual Analyses**: {self.config['analysis_settings']['individual_analyses']}\n\n")
            
            f.write("## Visualizations Generated\n\n")
            f.write("- `experiment_fitting_analysis.png`: Box plots with formula-based fitted lines and ±5% confidence bands\n\n")
        
        print(f"📄 Summary report saved to: {report_path}")
    
    def plot_parameter_constraint_analysis(self):
        """Generate parameter constraint analysis with new BPM-based constraints."""
        print("📊 Generating parameter constraint analysis...")
        
        constraint_config = self.config.get('constraint_analysis', {})
        bpm_list = constraint_config.get('bpm_list', [60, 80, 100, 120, 140, 160])
        percentile = constraint_config.get('percentile', 95)
        generation_frames = constraint_config.get('generation_frames', [1, 3, 5, 7, 9, 11, 13, 15])
        inference_intervals = constraint_config.get('inference_intervals', [1, 2, 3, 4, 5, 6, 7, 8])
        
        n_experiments = len(self.experiments_data)
        n_bpm = len(bpm_list)
        
        # Create research-quality layout using GridSpec for better control
        from matplotlib.gridspec import GridSpec
        
        # Calculate figure size for research publication
        subplot_size = 2.8  # Size of each subplot in inches
        fig_width = n_bpm * subplot_size + 1.5  # Extra space for legend
        fig_height = n_experiments * subplot_size + 1.0  # Extra space for titles
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Create GridSpec with tight spacing - no colorbar column needed
        gs = GridSpec(n_experiments, n_bpm, figure=fig, 
                     hspace=0.15, wspace=0,  # Tight spacing
                     left=0.08, right=0.95, top=0.90, bottom=0.18)
        
        # Create subplots with shared axes
        axes = []
        for exp_idx in range(n_experiments):
            exp_axes = []
            for bpm_idx in range(n_bpm):
                if exp_idx == 0 and bpm_idx == 0:
                    # First subplot - no sharing
                    ax = fig.add_subplot(gs[exp_idx, bpm_idx])
                    first_ax = ax
                elif exp_idx == 0:
                    # First row - share y-axis with first subplot
                    ax = fig.add_subplot(gs[exp_idx, bpm_idx], sharey=first_ax)
                elif bpm_idx == 0:
                    # First column - share x-axis with first subplot
                    ax = fig.add_subplot(gs[exp_idx, bpm_idx], sharex=first_ax)
                else:
                    # All others - share both axes with first subplot
                    ax = fig.add_subplot(gs[exp_idx, bpm_idx], sharex=first_ax, sharey=first_ax)
                exp_axes.append(ax)
            axes.append(exp_axes)
        
        for exp_idx, (formal_name, exp_info) in enumerate(self.experiments_data.items()):
            exp_data = exp_info['data']
            
            # Calculate percentile data for this experiment
            percentile_data = self._calculate_experiment_percentiles(exp_data, percentile, generation_frames)
            
            for bpm_idx, bpm in enumerate(bpm_list):
                ax = axes[exp_idx][bpm_idx]
                
                # Calculate tau_tick: time per tick in ms
                tau_tick = 15000 / bpm  # ms/tick
                
                # Create constraint matrix
                matrix = self._create_constraint_matrix_bpm(
                    percentile_data, tau_tick, inference_intervals, generation_frames, exp_info
                )
                
                # Plot heatmap with square cells
                im = ax.imshow(matrix, cmap=self._get_constraint_colormap(), 
                              aspect='equal', origin='lower', vmin=0, vmax=3)
                
                # Set labels and ticks with detailed information
                # Add BPM title only for top row
                if exp_idx == 0:
                    ax.set_title(f'BPM {bpm}', fontsize=14, fontweight='bold', pad=10)
                
                # Set ticks to show actual values with coordinate mapping
                x_positions = range(len(generation_frames))
                y_positions = range(len(inference_intervals))
                ax.set_xticks(x_positions)
                ax.set_xticklabels([str(gl) for gl in generation_frames], fontsize=12)
                ax.set_yticks(y_positions)
                ax.set_yticklabels([str(ii) for ii in inference_intervals], fontsize=12)
                
                # Add grid lines to separate cells but remove tick marks
                ax.set_xticks(np.arange(-0.5, len(generation_frames), 1), minor=True)
                ax.set_yticks(np.arange(-0.5, len(inference_intervals), 1), minor=True)
                # ax.grid(which='major', b=False)
                ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
                ax.tick_params(which='minor', size=0, width=0)  # Hide minor tick marks completely
                ax.tick_params(which='major', size=0, width=0, gridOn=False)

                # Add constraint status annotations
                # for i in range(len(inference_intervals)):
                #     for j in range(len(generation_frames)):
                #         if matrix[i, j] == 1:  # C1 violated
                #             ax.text(j, i, 'C1', ha='center', va='center', 
                #                    color='white', fontsize=7, fontweight='bold')
                #         elif matrix[i, j] == 2:  # C2 violated
                #             ax.text(j, i, 'C2', ha='center', va='center', 
                #                    color='white', fontsize=7, fontweight='bold')
                #         elif matrix[i, j] == 3:  # Both violated
                #             ax.text(j, i, 'C1+C2', ha='center', va='center', 
                #                    color='white', fontsize=6, fontweight='bold')
                
                # Overlay constraint boundary traces
                self._plot_constraint_boundaries(ax, exp_info['config'], tau_tick, inference_intervals)
            # Add experiment name on the left
            axes[exp_idx][0].text(-0.15, 0.5, formal_name, rotation=90, 
                                 transform=axes[exp_idx][0].transAxes, 
                                 fontsize=12, fontweight='bold', ha='center', va='center')
        
        # Add shared axis labels
        fig.text(0.5, 0.12, 'Generation Length (Frames)', ha='center', va='bottom', 
                fontsize=14, fontweight='bold')
        fig.text(0.04, 0.5, 'Inference Interval (Ticks)', ha='left', va='center', 
                rotation=90, fontsize=14, fontweight='bold')
        
        # Add main title
        # fig.suptitle('Constraint Analysis over Combinations of GL vs I across Settings and BPMs', 
        #             fontsize=16, fontweight='bold', y=0.96)
        
        # Create comprehensive legend combining constraint status and boundary lines
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        
        legend_elements = [
            # Constraint status patches
            Patch(facecolor='#2E7D32', label='Valid'),
            Patch(facecolor='#FF9800', label='C1 Violated'), 
            Patch(facecolor='#F44336', label='C2 Violated'),
            Patch(facecolor='#8B0000', label='Both Violated'),
            # Boundary lines
            Line2D([0], [0], color='black', linewidth=2, linestyle='-', label='C1 Boundary'),
            Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='C2 Boundary')
        ]
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.94), 
                  ncol=6, fontsize=12, frameon=True, fancybox=True, framealpha=0.9, edgecolor='black')
        
        # Save plot
        output_path = self.plots_dir / "parameter_constraint_analysis_bpm.png"
        output_path_pdf = output_path.with_suffix('.pdf')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Parameter constraint analysis saved to: {output_path}")
        
        # Also create individual plots for each experiment
        self._plot_individual_experiment_constraints(bpm_list, percentile, generation_frames, inference_intervals)
    
    def _calculate_experiment_percentiles(self, exp_data, percentile, generation_frames):
        """Calculate percentile data for an experiment at specific generation frames."""
        percentile_data = {}
        
        for gl in generation_frames:
            gl_data = exp_data[exp_data['generation_length'] == gl]
            if len(gl_data) > 0:
                rtt_percentile = gl_data['round_trip_time'].quantile(percentile / 100) * 1000  # Convert to ms
                percentile_data[gl] = rtt_percentile
        
        return percentile_data
    
    def _create_constraint_matrix_bpm(self, percentile_data, tau_tick, inference_intervals, generation_frames, exp_info=None):
        """Create constraint matrix using new BPM-based constraints."""
        matrix = np.full((len(inference_intervals), len(generation_frames)), -1, dtype=int)
        
        # Check if we should use formula-based approach
        constraint_config = self.config.get('constraint_analysis', {})
        use_formula_boundaries = constraint_config.get('use_formula_boundaries', False)
        use_formula_95 = constraint_config.get('use_formula_95', False)
        
        formula_type_desc = "95% formula" if use_formula_95 else "main formula" 
        print(f"   🔧 Using formula-based boundaries: {use_formula_boundaries} ({formula_type_desc})")
        
        if use_formula_boundaries and exp_info is not None:
            # Use formula-based boundary approach with experiment-specific formula
            print("   📊 Applying formula-based constraint evaluation...")
            matrix = self._create_constraint_matrix_formula_based(tau_tick, inference_intervals, generation_frames, exp_info)
        else:
            # Use empirical data approach (default)
            print("   📈 Applying empirical data constraint evaluation...")
            for i, inference_interval in enumerate(inference_intervals):
                for j, gl in enumerate(generation_frames):
                    if gl in percentile_data:
                        rt_ms = percentile_data[gl]
                        
                        # New constraint calculations
                        # Constraint 1: ceil(RT / tau_tick) <= inference_interval
                        required_ticks = np.ceil(rt_ms / tau_tick)
                        constraint1_satisfied = required_ticks <= inference_interval
                        
                        # Constraint 2: GL >= ceil(RT / tau_tick) + I
                        constraint2_satisfied = gl >= required_ticks + inference_interval
                        
                        # Determine constraint status
                        if constraint1_satisfied and constraint2_satisfied:
                            matrix[i, j] = 0  # Valid (both satisfied)
                        elif not constraint1_satisfied and constraint2_satisfied:
                            matrix[i, j] = 1  # Only constraint 1 violated
                        elif constraint1_satisfied and not constraint2_satisfied:
                            matrix[i, j] = 2  # Only constraint 2 violated
                        else:
                            matrix[i, j] = 3  # Both violated
        
        return matrix
    
    def _get_constraint_colormap(self):
        """Create custom colormap for constraint status."""
        from matplotlib.colors import ListedColormap
        colors = ['#2E7D32', '#FF9800', '#F44336', '#8B0000']  # Green, Orange, Red, Dark Red
        return ListedColormap(colors)
    
    def _create_constraint_matrix_formula_based(self, tau_tick, inference_intervals, generation_frames, exp_info):
        """Create constraint matrix using formula boundary lines approach."""
        matrix = np.full((len(inference_intervals), len(generation_frames)), -1, dtype=int)
        
        # Check if we should use the 95% formula instead of main formula
        constraint_config = self.config.get('constraint_analysis', {})
        use_formula_95 = constraint_config.get('use_formula_95', False)
        
        # Use the specific experiment's formula (main or 95%)
        if use_formula_95 and 'formula_95' in exp_info['config']:
            formula_config = exp_info['config']['formula_95']
            formula_type = "95% formula"
        else:
            formula_config = exp_info['config']['formula']
            formula_type = "main formula"
        
        experiment_name = exp_info['data']['formal_name'].iloc[0] if 'formal_name' in exp_info['data'] else "Unknown"
        print(f"   🧮 Using {experiment_name} {formula_type}: RT = {formula_config.get('a', 0):.4f}×GL² + {formula_config.get('b', 0):.4f}×GL + {formula_config.get('c', 0):.4f}")
        
        for i, inference_interval in enumerate(inference_intervals):
            for j, gl in enumerate(generation_frames):
                # Check if point (GL, I) is valid based on formula boundaries
                constraint1_satisfied, constraint2_satisfied = self._check_formula_constraints(
                    gl, inference_interval, formula_config, tau_tick
                )
                
                # Determine constraint status
                if constraint1_satisfied and constraint2_satisfied:
                    matrix[i, j] = 0  # Valid (both satisfied)
                elif not constraint1_satisfied and constraint2_satisfied:
                    matrix[i, j] = 1  # Only constraint 1 violated
                elif constraint1_satisfied and not constraint2_satisfied:
                    matrix[i, j] = 2  # Only constraint 2 violated
                else:
                    matrix[i, j] = 3  # Both violated
        
        print(f"   📊 Formula-based matrix calculated using {formula_type}. Valid cells: {np.sum(matrix == 0)}, Total cells: {np.sum(matrix >= 0)}")
        return matrix
    
    def _check_formula_constraints(self, gl, inference_interval, formula_config, tau_tick):
        """Check if a (GL, I) point satisfies constraints using formula boundaries."""
        
        a = formula_config.get('a', 0)
        b = formula_config.get('b', 0)
        c = formula_config.get('c', 0)
        
        # Calculate RT using the formula for this GL (already in ms based on YAML coefficients)
        rt_formula = self.calculate_formula_values(np.array([gl]), formula_config)[0]
        
        # Debug: print detailed calculation for specific point
        if gl == 5 and inference_interval == 2:
            print(f"   🔍 DETAILED DEBUG for GL=5, I=2:")
            print(f"   🔍 RT_formula = {rt_formula:.2f}ms")
            print(f"   🔍 tau_tick = {tau_tick:.2f}ms")
        
        # Debug: print a few sample calculations
        if gl == 1 and inference_interval == 1:
            print(f"   🔍 Debug sample (GL=1, I=1): RT_formula={rt_formula:.2f}ms, tau_tick={tau_tick:.2f}ms")
        
        try:
            # Constraint 1: ceil(RT / tau_tick) <= I
            # This means the inference interval must be long enough to avoid overlap
            required_ticks = np.ceil(rt_formula / tau_tick)
            constraint1_satisfied = required_ticks <= inference_interval
            
            # Constraint 2: GL >= ceil(RT / tau_tick) + I
            # This means generated music must arrive before it's needed
            # The generation length must be at least the required processing time plus inference interval
            constraint2_satisfied = gl >= required_ticks + inference_interval
            
            # Debug: print more details for both sample cases
            if (gl == 1 and inference_interval == 1) or (gl == 5 and inference_interval == 2):
                print(f"   🔍 Debug GL={gl}, I={inference_interval}: required_ticks={required_ticks}")
                print(f"   🔍 C1: ceil({rt_formula:.2f}/{tau_tick:.2f}) = {required_ticks} <= {inference_interval} = {constraint1_satisfied}")
                print(f"   🔍 C2: {gl} >= {required_ticks} + {inference_interval} = {gl} >= {required_ticks + inference_interval} = {constraint2_satisfied}")
                print(f"   🔍 Final result: C1={constraint1_satisfied}, C2={constraint2_satisfied}")
            
        except (ValueError, ZeroDivisionError):
            constraint1_satisfied = False
            constraint2_satisfied = False
            print(f"   ⚠️ Error calculating constraints for GL={gl}, I={inference_interval}")
        
        return constraint1_satisfied, constraint2_satisfied
    
    def _plot_individual_experiment_constraints(self, bpm_list, percentile, generation_frames, inference_intervals):
        """Create individual constraint plots for each experiment."""
        for formal_name, exp_info in self.experiments_data.items():
            exp_data = exp_info['data']
            
            # Calculate percentile data for this experiment
            percentile_data = self._calculate_experiment_percentiles(exp_data, percentile, generation_frames)
            
            # Create figure with one row of BPM heatmaps
            fig, axes = plt.subplots(1, len(bpm_list), figsize=(4*len(bpm_list), 4))
            if len(bpm_list) == 1:
                axes = [axes]
            
            for bpm_idx, bpm in enumerate(bpm_list):
                ax = axes[bpm_idx]
                
                # Calculate tau_tick
                tau_tick = 15000 / bpm
                
                # Create constraint matrix
                matrix = self._create_constraint_matrix_bpm(
                    percentile_data, tau_tick, inference_intervals, generation_frames, exp_info
                )
                
                # Plot heatmap with square cells
                im = ax.imshow(matrix, cmap=self._get_constraint_colormap(), 
                              aspect='equal', origin='lower', vmin=0, vmax=3)
                
                # Set labels and ticks with detailed information
                ax.set_xlabel('Generation Length (Frames)', fontsize=12)
                if bpm_idx == 0:
                    ax.set_ylabel('Inference Interval (Ticks)', fontsize=12)
                ax.set_title(f'BPM: {bpm}\nτ = {tau_tick:.1f} ms/tick', fontsize=12)
                
                # Set ticks to show actual values with coordinate mapping
                x_positions = range(len(generation_frames))
                y_positions = range(len(inference_intervals))
                ax.set_xticks(x_positions)
                ax.set_xticklabels([str(gl) for gl in generation_frames], fontsize=10)
                ax.set_yticks(y_positions)
                ax.set_yticklabels([str(ii) for ii in inference_intervals], fontsize=10)
                
                # Add grid lines to separate cells but remove tick marks
                ax.set_xticks(np.arange(-0.5, len(generation_frames), 1), minor=True)
                ax.set_yticks(np.arange(-0.5, len(inference_intervals), 1), minor=True)
                ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
                ax.tick_params(which='minor', size=0, width=0)  # Hide minor tick marks completely
                ax.tick_params(which='major', size=0, width=0)  # Hide major tick marks completely
                
                # Add constraint status annotations
                for i in range(len(inference_intervals)):
                    for j in range(len(generation_frames)):
                        if matrix[i, j] == 1:  # C1 violated
                            ax.text(j, i, 'C1', ha='center', va='center', 
                                   color='white', fontsize=7, fontweight='bold')
                        elif matrix[i, j] == 2:  # C2 violated
                            ax.text(j, i, 'C2', ha='center', va='center', 
                                   color='white', fontsize=7, fontweight='bold')
                        elif matrix[i, j] == 3:  # Both violated
                            ax.text(j, i, 'C1+C2', ha='center', va='center', 
                                   color='white', fontsize=6, fontweight='bold')
                
                # Overlay constraint boundary traces
                self._plot_constraint_boundaries(ax, exp_info['config'], tau_tick, inference_intervals)
            
            plt.suptitle(f'Parameter Constraint Analysis: {formal_name}', fontsize=16, fontweight='bold')
            
            # Adjust subplot spacing to make room for colorbar
            plt.subplots_adjust(right=0.85)
            
            # Add colorbar positioned better
            cbar = plt.colorbar(im, ax=axes, shrink=0.7, aspect=15)
            cbar.set_ticks([0, 1, 2, 3])
            cbar.set_ticklabels(['Valid', 'C1 Violated', 'C2 Violated', 'Both Violated'])
            cbar.set_label('Constraint Status', fontsize=12)
            
            # Save individual plot
            safe_name = formal_name.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
            output_path = self.plots_dir / f"constraint_analysis_{safe_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"✅ Individual constraint analysis for {formal_name} saved to: {output_path}")
    
    def _plot_constraint_boundaries(self, ax, exp_config, tau_tick, inference_intervals):
        """Plot constraint boundary traces on the heatmap."""
        formula_config = exp_config['formula']
        a = formula_config.get('a', 0)
        b = formula_config.get('b', 0) 
        c = formula_config.get('c', 0)
        t = tau_tick  # tau_tick as 't' in the formulas
        
        # Create continuous traces that extend beyond edges for clipping
        # Use fine resolution for smooth lines with extended range
        I_range = np.linspace(0, 10, 200)  # Extended range for edge coverage
        
        x_coords_1 = []
        y_coords_1 = []
        x_coords_2 = []
        y_coords_2 = []
        
        for I in I_range:
            y_grid = I - 1  # Convert I to grid coordinate
            
            # Formula 1: GL = (-b + sqrt(b² - 4(a)(c - I*tau_tick))) / (2a)
            try:
                discriminant1 = b**2 - 4*a*(c - I*tau_tick)
                if discriminant1 >= 0 and a != 0:
                    GL1 = (-b + np.sqrt(discriminant1)) / (2*a)
                    x_grid1 = (GL1 - 1) / 2  # Convert GL to grid coordinate
                    # Allow extended range for clipping
                    if -2 <= y_grid <= 10:
                        x_coords_1.append(x_grid1)
                        y_coords_1.append(y_grid)
            except (ValueError, ZeroDivisionError):
                pass
            
            # Formula 2: GL = (-(b-t) - sqrt((b-t)² - 4(a)(c + I*tau_tick))) / (2a)
            try:
                b_minus_t = b - t
                discriminant2 = b_minus_t**2 - 4*a*(c + I*tau_tick)
                if discriminant2 >= 0 and a != 0:
                    GL2 = (-b_minus_t - np.sqrt(discriminant2)) / (2*a)
                    x_grid2 = (GL2 - 1) / 2  # Convert GL to grid coordinate
                    # Allow extended range for clipping
                    if -2 <= y_grid <= 10:
                        x_coords_2.append(x_grid2)
                        y_coords_2.append(y_grid)
            except (ValueError, ZeroDivisionError):
                pass
        
        # Plot the traces with clipping to grid area and set axis limits
        ax.set_xlim(-0.5, 7.5)
        ax.set_ylim(-0.5, 7.5)
        
        if x_coords_1 and y_coords_1:
            ax.plot(x_coords_1, y_coords_1, 'k-', linewidth=2, alpha=0.8, label='C1 Boundary')
        
        if x_coords_2 and y_coords_2:
            ax.plot(x_coords_2, y_coords_2, 'k--', linewidth=2, alpha=0.8, label='C2 Boundary')
        
        # Don't add individual legends - will add one global legend later

    def run_individual_analyses(self):
        """Run individual analysis for each experiment."""
        print("📊 Running individual experiment analyses...")
        
        for formal_name, exp_info in self.experiments_data.items():
            print(f"   Analyzing {formal_name}...")
            exp_data = exp_info['data']
            
            # Create individual plots for this experiment
            self._plot_individual_rt_vs_gl(formal_name, exp_data, exp_info['config'])
            self._plot_individual_distributions(formal_name, exp_data)
            self._plot_individual_metrics_summary(formal_name, exp_data)
        
        # Create comparative plots
        self._plot_experiment_comparison()
        self._plot_generation_length_trends()
        
        print("✅ Individual analyses complete!")
    
    def _plot_individual_rt_vs_gl(self, formal_name, exp_data, exp_config):
        """Create individual RT vs GL plot for an experiment."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Get unique generation lengths
        gen_lengths = sorted(exp_data['generation_length'].unique())
        
        # Left plot: Box plots of RT vs GL
        box_data = []
        box_positions = []
        for gl in gen_lengths:
            gl_data = exp_data[exp_data['generation_length'] == gl]
            if len(gl_data) > 0:
                rtt_ms = gl_data['round_trip_time'] * 1000  # Convert to ms
                box_data.append(rtt_ms.values)
                box_positions.append(gl)
        
        if box_data:
            bp = ax1.boxplot(box_data, positions=box_positions, patch_artist=True, 
                           showfliers=True, widths=0.8)
            
            # Color boxes with experiment color
            color = exp_config['color']
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                patch.set_edgecolor('black')
        
        ax1.set_xlabel('Generation Length (Frames)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Round Trip Time (ms)', fontsize=14, fontweight='bold')
        ax1.set_title(f'RT vs GL Distribution: {formal_name}', fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Fitted line vs empirical data
        gl_range = np.linspace(min(gen_lengths), max(gen_lengths), 100)
        formula_config = exp_config['formula']
        
        # Calculate fitted line
        fitted_values = self.calculate_formula_values(gl_range, formula_config) * 1000  # Convert to ms
        
        # Plot empirical means
        gl_means = []
        rtt_means = []
        for gl in gen_lengths:
            gl_data = exp_data[exp_data['generation_length'] == gl]
            if len(gl_data) > 0:
                gl_means.append(gl)
                rtt_means.append(gl_data['round_trip_time'].mean() * 1000)
        
        ax2.scatter(gl_means, rtt_means, color=color, s=80, alpha=0.8, label='Empirical Mean', zorder=5)
        ax2.plot(gl_range, fitted_values, color='black', linewidth=2.5, label='Fitted Formula', zorder=3)
        
        # Add confidence bands
        upper_band = fitted_values * 1.05
        lower_band = fitted_values * 0.95
        ax2.fill_between(gl_range, lower_band, upper_band, color=color, alpha=0.2, label='±5% Band')
        
        ax2.set_xlabel('Generation Length (Frames)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Round Trip Time (ms)', fontsize=14, fontweight='bold')
        ax2.set_title(f'Formula Fit: {formal_name}', fontsize=16, fontweight='bold')
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        safe_name = formal_name.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
        output_path = self.plots_dir / f"individual_rt_vs_gl_{safe_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ RT vs GL plot saved: {output_path}")
    
    def _plot_individual_distributions(self, formal_name, exp_data):
        """Create distribution analysis plots for an experiment."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Get unique generation lengths
        gen_lengths = sorted(exp_data['generation_length'].unique())
        
        # Plot 1: RT distribution by GL
        for gl in gen_lengths:
            gl_data = exp_data[exp_data['generation_length'] == gl]
            if len(gl_data) > 0:
                rtt_ms = gl_data['round_trip_time'] * 1000
                axes[0].hist(rtt_ms, bins=20, alpha=0.6, label=f'GL {gl}', density=True)
        
        axes[0].set_xlabel('Round Trip Time (ms)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Density', fontsize=12, fontweight='bold')
        axes[0].set_title('RT Distribution by Generation Length', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Inference time vs GL if available
        if 'inference_duration' in exp_data.columns:
            inf_means = []
            inf_stds = []
            for gl in gen_lengths:
                gl_data = exp_data[exp_data['generation_length'] == gl]
                if len(gl_data) > 0:
                    inf_means.append(gl_data['inference_duration'].mean() * 1000)
                    inf_stds.append(gl_data['inference_duration'].std() * 1000)
            
            axes[1].errorbar(gen_lengths, inf_means, yerr=inf_stds, marker='o', linewidth=2, markersize=8)
            axes[1].set_xlabel('Generation Length (Frames)', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Inference Duration (ms)', fontsize=12, fontweight='bold')
            axes[1].set_title('Inference Time vs Generation Length', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Percentile analysis
        percentiles = [50, 75, 90, 95, 99]
        for p in percentiles:
            p_values = []
            for gl in gen_lengths:
                gl_data = exp_data[exp_data['generation_length'] == gl]
                if len(gl_data) > 0:
                    p_val = gl_data['round_trip_time'].quantile(p/100) * 1000
                    p_values.append(p_val)
            axes[2].plot(gen_lengths, p_values, marker='o', linewidth=2, label=f'P{p}')
        
        axes[2].set_xlabel('Generation Length (Frames)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Round Trip Time (ms)', fontsize=12, fontweight='bold')
        axes[2].set_title('Percentile Analysis', fontsize=14, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Sample count by GL
        sample_counts = []
        for gl in gen_lengths:
            gl_data = exp_data[exp_data['generation_length'] == gl]
            sample_counts.append(len(gl_data))
        
        bars = axes[3].bar(gen_lengths, sample_counts, alpha=0.7)
        axes[3].set_xlabel('Generation Length (Frames)', fontsize=12, fontweight='bold')
        axes[3].set_ylabel('Sample Count', fontsize=12, fontweight='bold')
        axes[3].set_title('Sample Count by Generation Length', fontsize=14, fontweight='bold')
        axes[3].grid(True, alpha=0.3, axis='y')
        
        # Add count labels on bars
        for bar, count in zip(bars, sample_counts):
            axes[3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle(f'Distribution Analysis: {formal_name}', fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        safe_name = formal_name.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
        output_path = self.plots_dir / f"individual_distributions_{safe_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ Distribution analysis saved: {output_path}")
    
    def _plot_individual_metrics_summary(self, formal_name, exp_data):
        """Create metrics summary table for an experiment."""
        gen_lengths = sorted(exp_data['generation_length'].unique())
        
        # Calculate metrics for each GL
        metrics_data = []
        for gl in gen_lengths:
            gl_data = exp_data[exp_data['generation_length'] == gl]
            if len(gl_data) > 0:
                rtt = gl_data['round_trip_time'] * 1000  # Convert to ms
                
                metrics = {
                    'Generation Length': gl,
                    'Samples': len(gl_data),
                    'Mean RT (ms)': rtt.mean(),
                    'Std RT (ms)': rtt.std(),
                    'P50 (ms)': rtt.quantile(0.5),
                    'P95 (ms)': rtt.quantile(0.95),
                    'P99 (ms)': rtt.quantile(0.99),
                    'Min (ms)': rtt.min(),
                    'Max (ms)': rtt.max()
                }
                
                if 'inference_duration' in gl_data.columns:
                    inf = gl_data['inference_duration'] * 1000
                    metrics.update({
                        'Mean Inf (ms)': inf.mean(),
                        'P95 Inf (ms)': inf.quantile(0.95)
                    })
                
                metrics_data.append(metrics)
        
        # Create metrics table plot
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.axis('tight')
            ax.axis('off')
            
            # Create table
            table_data = []
            headers = list(metrics_df.columns)
            table_data.append(headers)
            
            for _, row in metrics_df.iterrows():
                formatted_row = []
                for col in headers:
                    if col in ['Samples', 'Generation Length']:
                        formatted_row.append(f"{row[col]:.0f}")
                    else:
                        formatted_row.append(f"{row[col]:.2f}")
                table_data.append(formatted_row)
            
            table = ax.table(cellText=table_data[1:], colLabels=headers, 
                           cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 2)
            
            # Style header
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Alternate row colors
            for i in range(1, len(table_data)):
                for j in range(len(headers)):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f0f0f0')
            
            plt.title(f'Performance Metrics Summary: {formal_name}', 
                     fontsize=16, fontweight='bold', pad=20)
            
            # Save plot
            safe_name = formal_name.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
            output_path = self.plots_dir / f"individual_metrics_{safe_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"   ✅ Metrics summary saved: {output_path}")
    
    def _plot_experiment_comparison(self):
        """Create comparative analysis across experiments."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Plot 1: Mean RT comparison
        for formal_name, exp_info in self.experiments_data.items():
            exp_data = exp_info['data']
            color = exp_info['config']['color']
            
            gen_lengths = sorted(exp_data['generation_length'].unique())
            means = []
            stds = []
            
            for gl in gen_lengths:
                gl_data = exp_data[exp_data['generation_length'] == gl]
                if len(gl_data) > 0:
                    means.append(gl_data['round_trip_time'].mean() * 1000)
                    stds.append(gl_data['round_trip_time'].std() * 1000)
            
            ax1.errorbar(gen_lengths, means, yerr=stds, label=formal_name, 
                        marker='o', linewidth=2.5, markersize=8, color=color)
        
        ax1.set_xlabel('Generation Length (Frames)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Mean Round Trip Time (ms)', fontsize=14, fontweight='bold')
        ax1.set_title('Mean RT Comparison', fontsize=16, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: 95th percentile comparison
        for formal_name, exp_info in self.experiments_data.items():
            exp_data = exp_info['data']
            color = exp_info['config']['color']
            
            gen_lengths = sorted(exp_data['generation_length'].unique())
            p95_values = []
            
            for gl in gen_lengths:
                gl_data = exp_data[exp_data['generation_length'] == gl]
                if len(gl_data) > 0:
                    p95_values.append(gl_data['round_trip_time'].quantile(0.95) * 1000)
            
            ax2.plot(gen_lengths, p95_values, label=formal_name, 
                    marker='s', linewidth=2.5, markersize=8, color=color)
        
        ax2.set_xlabel('Generation Length (Frames)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('95th Percentile RT (ms)', fontsize=14, fontweight='bold')
        ax2.set_title('P95 RT Comparison', fontsize=16, fontweight='bold')
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Fitted formula comparison
        gl_range = np.linspace(1, 15, 100)
        for formal_name, exp_info in self.experiments_data.items():
            formula_config = exp_info['config']['formula']
            color = exp_info['config']['color']
            
            fitted_values = self.calculate_formula_values(gl_range, formula_config) * 1000
            ax3.plot(gl_range, fitted_values, label=formal_name, 
                    linewidth=3, color=color)
        
        ax3.set_xlabel('Generation Length (Frames)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Fitted RT (ms)', fontsize=14, fontweight='bold')
        ax3.set_title('Formula Comparison', fontsize=16, fontweight='bold')
        ax3.legend(fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Sample count comparison
        for formal_name, exp_info in self.experiments_data.items():
            exp_data = exp_info['data']
            color = exp_info['config']['color']
            
            gen_lengths = sorted(exp_data['generation_length'].unique())
            counts = []
            
            for gl in gen_lengths:
                gl_data = exp_data[exp_data['generation_length'] == gl]
                counts.append(len(gl_data))
            
            ax4.plot(gen_lengths, counts, label=formal_name, 
                    marker='d', linewidth=2.5, markersize=8, color=color)
        
        ax4.set_xlabel('Generation Length (Frames)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Sample Count', fontsize=14, fontweight='bold')
        ax4.set_title('Sample Count Comparison', fontsize=16, fontweight='bold')
        ax4.legend(fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.plots_dir / "experiment_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Experiment comparison saved: {output_path}")
    
    def _plot_generation_length_trends(self):
        """Create generation length scaling analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        metrics = [
            ('round_trip_time', 'Round Trip Time (ms)', 1000),
            ('inference_duration', 'Inference Duration (ms)', 1000) if any('inference_duration' in exp_info['data'].columns for exp_info in self.experiments_data.values()) else None
        ]
        
        # Filter out None metrics
        metrics = [m for m in metrics if m is not None]
        
        for i, (metric, ylabel, scale_factor) in enumerate(metrics):
            if i < len(axes):
                ax = axes[i]
                
                for formal_name, exp_info in self.experiments_data.items():
                    exp_data = exp_info['data']
                    color = exp_info['config']['color']
                    
                    if metric in exp_data.columns:
                        gen_lengths = sorted(exp_data['generation_length'].unique())
                        means = []
                        
                        for gl in gen_lengths:
                            gl_data = exp_data[exp_data['generation_length'] == gl]
                            if len(gl_data) > 0:
                                means.append(gl_data[metric].mean() * scale_factor)
                        
                        ax.plot(gen_lengths, means, label=formal_name, 
                               marker='o', linewidth=2.5, markersize=8, color=color)
                
                ax.set_xlabel('Generation Length (Frames)', fontsize=14, fontweight='bold')
                ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
                ax.set_title(f'{ylabel} Scaling', fontsize=16, fontweight='bold')
                ax.legend(fontsize=12)
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.plots_dir / "generation_length_trends.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Generation length trends saved: {output_path}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="YAML-based StreamMUSE Bulk Analysis")
    parser.add_argument("config_file", help="Path to YAML configuration file")
    parser.add_argument("--output_dir", help="Override output directory from config")
    
    args = parser.parse_args()
    
    # Initialize analysis engine
    try:
        engine = YAMLBulkAnalysisEngine(args.config_file)
        
        # Override output directory if specified
        if args.output_dir:
            engine.output_dir = Path(args.output_dir)
            engine.output_dir.mkdir(parents=True, exist_ok=True)
            engine.plots_dir = engine.output_dir / 'plots'
            engine.plots_dir.mkdir(exist_ok=True)
        
        # Load experiments and run analysis
        engine.load_experiments()
        engine.plot_experiment_fitting()
        engine.plot_parameter_constraint_analysis()
        
        # Add individual analysis if enabled
        if engine.config['analysis_settings'].get('individual_analyses', False):
            engine.run_individual_analyses()
            
        engine.generate_summary_report()
        
        print(f"\n✅ YAML-based analysis complete! Results saved to: {engine.output_dir}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()