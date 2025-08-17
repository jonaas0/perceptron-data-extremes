"""
Result analysis and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

from ..models import ExperimentResult, ConvergenceStatus
from .results_collector import ResultsCollector


logger = logging.getLogger(__name__)


class AnalysisEngine:
    """Generate visualizations and statistical comparisons for experimental results."""
    
    def __init__(self, results_collector: ResultsCollector, output_dir: str = "outputs"):
        """
        Initialize analysis engine with results collector.
        
        Args:
            results_collector: ResultsCollector containing experimental data
            output_dir: Directory to save generated plots and analysis
        """
        self.results_collector = results_collector
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        logger.info(f"AnalysisEngine initialized with {results_collector.get_results_count()} results")
    
    def _prepare_dataframe(self) -> pd.DataFrame:
        """
        Prepare a pandas DataFrame from experimental results for analysis.
        
        Returns:
            DataFrame with experimental data
        """
        results = self.results_collector.get_all_results()
        
        if not results:
            raise ValueError("No experimental results available for analysis")
        
        data = []
        for result in results:
            row = {
                'experiment_id': result.experiment_id,
                'dataset_size': result.dataset_size,
                'extreme_value_ratio': result.extreme_value_ratio,
                'repetition': result.repetition,
                'final_weight': result.final_weights[0] if result.final_weights else np.nan,
                'final_bias': result.final_bias,
                'training_iterations': result.training_iterations,
                'convergence_status': result.convergence_status.value,
                'converged': result.convergence_status == ConvergenceStatus.CONVERGED,
                'execution_time': result.execution_time,
                'dataset_mean': result.dataset_properties.actual_mean,
                'extreme_value_count': result.dataset_properties.extreme_value_count,
                'validation_passed': result.dataset_properties.statistical_validation_passed
            }
            
            # Add training metrics
            for metric_name, metric_value in result.training_metrics.items():
                row[f'training_{metric_name}'] = metric_value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def create_weight_distribution_plots(self) -> List[str]:
        """
        Create visualizations comparing weight distributions across conditions.
        
        Returns:
            List of generated plot file paths
        """
        df = self._prepare_dataframe()
        plot_files = []
        
        # 1. Weight distribution by extreme value ratio
        plt.figure(figsize=(12, 8))
        
        # Create subplot for each dataset size
        dataset_sizes = sorted(df['dataset_size'].unique())
        n_sizes = len(dataset_sizes)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, size in enumerate(dataset_sizes):
            if i < len(axes):
                ax = axes[i]
                size_data = df[df['dataset_size'] == size]
                
                # Box plot of weights by extreme value ratio
                extreme_ratios = sorted(size_data['extreme_value_ratio'].unique())
                weight_data = [
                    size_data[size_data['extreme_value_ratio'] == ratio]['final_weight'].dropna()
                    for ratio in extreme_ratios
                ]
                
                bp = ax.boxplot(weight_data, labels=[f"{int(r*100)}%" for r in extreme_ratios])
                ax.set_title(f'Weight Distribution - Dataset Size {size}')
                ax.set_xlabel('Extreme Value Ratio')
                ax.set_ylabel('Final Weight')
                ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(dataset_sizes), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        weight_dist_path = self.output_dir / "weight_distributions_by_condition.png"
        plt.savefig(weight_dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(weight_dist_path))
        
        # 2. Weight evolution heatmap
        plt.figure(figsize=(10, 8))
        
        # Create pivot table for heatmap
        pivot_data = df.groupby(['dataset_size', 'extreme_value_ratio'])['final_weight'].mean().unstack()
        
        sns.heatmap(
            pivot_data, 
            annot=True, 
            fmt='.3f', 
            cmap='viridis',
            cbar_kws={'label': 'Average Final Weight'}
        )
        plt.title('Average Final Weights by Dataset Size and Extreme Value Ratio')
        plt.xlabel('Extreme Value Ratio')
        plt.ylabel('Dataset Size')
        
        heatmap_path = self.output_dir / "weight_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(heatmap_path))
        
        logger.info(f"Generated {len(plot_files)} weight distribution plots")
        return plot_files

    def create_bias_distribution_plots(self) -> List[str]:
        """
        Create visualizations comparing bias distributions across conditions.
        
        Returns:
            List of generated plot file paths
        """
        df = self._prepare_dataframe()
        plot_files = []
        
        # 1. Bias distribution by extreme value ratio
        plt.figure(figsize=(12, 8))
        
        # Create subplot for each dataset size
        dataset_sizes = sorted(df['dataset_size'].unique())
        n_sizes = len(dataset_sizes)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, size in enumerate(dataset_sizes):
            if i < len(axes):
                ax = axes[i]
                size_data = df[df['dataset_size'] == size]
                
                # Box plot of bias by extreme value ratio
                extreme_ratios = sorted(size_data['extreme_value_ratio'].unique())
                bias_data = [
                    size_data[size_data['extreme_value_ratio'] == ratio]['final_bias'].dropna()
                    for ratio in extreme_ratios
                ]
                
                bp = ax.boxplot(bias_data, labels=[f"{int(r*100)}%" for r in extreme_ratios])
                ax.set_title(f'Bias Distribution - Dataset Size {size}')
                ax.set_xlabel('Extreme Value Ratio')
                ax.set_ylabel('Final Bias')
                ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Expected (1.0)')
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        # Remove empty subplots
        for i in range(len(dataset_sizes), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        bias_dist_path = self.output_dir / "bias_distributions_by_condition.png"
        plt.savefig(bias_dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(bias_dist_path))
        
        # 2. Bias evolution heatmap
        plt.figure(figsize=(10, 8))
        
        # Create pivot table for heatmap
        pivot_data = df.groupby(['dataset_size', 'extreme_value_ratio'])['final_bias'].mean().unstack()
        
        sns.heatmap(
            pivot_data, 
            annot=True, 
            fmt='.3f', 
            cmap='RdYlBu_r',
            center=1.0,  # Center the colormap at expected bias value
            cbar_kws={'label': 'Average Final Bias'}
        )
        plt.title('Average Final Bias by Dataset Size and Extreme Value Ratio')
        plt.xlabel('Extreme Value Ratio')
        plt.ylabel('Dataset Size')
        
        bias_heatmap_path = self.output_dir / "bias_heatmap.png"
        plt.savefig(bias_heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(bias_heatmap_path))
        
        logger.info(f"Generated {len(plot_files)} bias distribution plots")
        return plot_files
    
    def compare_extreme_value_effects(self) -> Dict:
        """
        Provide statistical comparisons between different extreme value proportions.
        
        Returns:
            Dictionary with statistical comparison results
        """
        df = self._prepare_dataframe()
        
        comparisons = {
            'weight_analysis': {},
            'convergence_analysis': {},
            'training_time_analysis': {},
            'statistical_tests': {}
        }
        
        # Weight analysis by extreme value ratio
        weight_stats = df.groupby('extreme_value_ratio')['final_weight'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(4)
        comparisons['weight_analysis'] = weight_stats.to_dict('index')
        
        # Bias analysis by extreme value ratio
        bias_stats = df.groupby('extreme_value_ratio')['final_bias'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(4)
        comparisons['bias_analysis'] = bias_stats.to_dict('index')
        
        # Convergence analysis
        convergence_stats = df.groupby('extreme_value_ratio')['converged'].agg([
            'count', 'sum', 'mean'
        ])
        convergence_stats.columns = ['total_experiments', 'converged_count', 'convergence_rate']
        comparisons['convergence_analysis'] = convergence_stats.to_dict('index')
        
        # Training time analysis
        time_stats = df.groupby('extreme_value_ratio')['execution_time'].agg([
            'mean', 'std', 'min', 'max'
        ]).round(3)
        comparisons['training_time_analysis'] = time_stats.to_dict('index')
        
        # Dataset size effects on weights
        size_effects_weights = df.groupby('dataset_size')['final_weight'].agg([
            'count', 'mean', 'std'
        ]).round(4)
        comparisons['dataset_size_effects_weights'] = size_effects_weights.to_dict('index')
        
        # Dataset size effects on weights
        size_effects = df.groupby('dataset_size')['final_weight'].agg([
            'count', 'mean', 'std'
        ]).round(4)
        comparisons['dataset_size_effects'] = size_effects.to_dict('index')
        
        logger.info("Generated statistical comparisons for extreme value effects")
        return comparisons
    
    def analyze_dataset_size_trends(self) -> Dict:
        """
        Analyze trends across different dataset sizes.
        
        Returns:
            Dictionary with trend analysis results
        """
        df = self._prepare_dataframe()
        
        trends = {
            'weight_trends': {},
            'convergence_trends': {},
            'stability_trends': {}
        }
        
        # Weight and bias trends by dataset size
        for size in sorted(df['dataset_size'].unique()):
            size_data = df[df['dataset_size'] == size]
            
            trends['weight_trends'][size] = {
                'mean_weight': size_data['final_weight'].mean(),
                'weight_std': size_data['final_weight'].std(),
                'weight_range': size_data['final_weight'].max() - size_data['final_weight'].min(),
                'mean_bias': size_data['final_bias'].mean(),
                'bias_std': size_data['final_bias'].std(),
                'bias_range': size_data['final_bias'].max() - size_data['final_bias'].min()
            }
            
            trends['convergence_trends'][size] = {
                'convergence_rate': size_data['converged'].mean(),
                'avg_iterations': size_data['training_iterations'].mean()
            }
            
            # Stability analysis (coefficient of variation)
            weight_cv = size_data['final_weight'].std() / size_data['final_weight'].mean() if size_data['final_weight'].mean() != 0 else 0
            bias_cv = size_data['final_bias'].std() / abs(size_data['final_bias'].mean()) if size_data['final_bias'].mean() != 0 else 0
            trends['stability_trends'][size] = {
                'weight_coefficient_of_variation': weight_cv,
                'bias_coefficient_of_variation': bias_cv,
                'execution_time_std': size_data['execution_time'].std()
            }
        
        logger.info("Generated dataset size trend analysis")
        return trends
    
    def create_summary_graphs(self) -> List[str]:
        """
        Generate summary graphs comparing extreme value proportions and dataset sizes.
        
        Returns:
            List of generated plot file paths
        """
        df = self._prepare_dataframe()
        plot_files = []
        
        # 1. Summary comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Average weights by extreme value ratio
        weight_by_ratio = df.groupby('extreme_value_ratio')['final_weight'].mean()
        ax1.bar(range(len(weight_by_ratio)), weight_by_ratio.values, 
                color='skyblue', alpha=0.7)
        ax1.set_xlabel('Extreme Value Ratio')
        ax1.set_ylabel('Average Final Weight')
        ax1.set_title('Average Final Weight by Extreme Value Ratio')
        ax1.set_xticks(range(len(weight_by_ratio)))
        ax1.set_xticklabels([f"{int(r*100)}%" for r in weight_by_ratio.index])
        ax1.grid(True, alpha=0.3)
        
        # Add expected weight line (y = 2x + 1, so weight should be ~2.0)
        ax1.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Expected Weight (2.0)')
        ax1.legend()
        
        # Plot 2: Average bias by extreme value ratio
        bias_by_ratio = df.groupby('extreme_value_ratio')['final_bias'].mean()
        ax2.bar(range(len(bias_by_ratio)), bias_by_ratio.values, 
                color='lightcoral', alpha=0.7)
        ax2.set_xlabel('Extreme Value Ratio')
        ax2.set_ylabel('Average Final Bias')
        ax2.set_title('Average Final Bias by Extreme Value Ratio')
        ax2.set_xticks(range(len(bias_by_ratio)))
        ax2.set_xticklabels([f"{int(r*100)}%" for r in bias_by_ratio.index])
        ax2.grid(True, alpha=0.3)
        
        # Add expected bias line (y = 2x + 1, so bias should be ~1.0)
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Expected Bias (1.0)')
        ax2.legend()
        
        # Plot 3: Convergence rates
        conv_by_ratio = df.groupby('extreme_value_ratio')['converged'].mean()
        ax3.bar(range(len(conv_by_ratio)), conv_by_ratio.values, 
                color='lightgreen', alpha=0.7)
        ax3.set_xlabel('Extreme Value Ratio')
        ax3.set_ylabel('Convergence Rate')
        ax3.set_title('Convergence Rate by Extreme Value Ratio')
        ax3.set_xticks(range(len(conv_by_ratio)))
        ax3.set_xticklabels([f"{int(r*100)}%" for r in conv_by_ratio.index])
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Training iterations
        iter_by_size = df.groupby('dataset_size')['training_iterations'].mean()
        ax4.bar(range(len(iter_by_size)), iter_by_size.values, 
                color='gold', alpha=0.7)
        ax4.set_xlabel('Dataset Size')
        ax4.set_ylabel('Average Training Iterations')
        ax4.set_title('Average Training Iterations by Dataset Size')
        ax4.set_xticks(range(len(iter_by_size)))
        ax4.set_xticklabels(iter_by_size.index)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        summary_path = self.output_dir / "experiment_summary_graphs.png"
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(summary_path))
        
        # 2. Detailed scatter plot
        plt.figure(figsize=(12, 8))
        
        # Scatter plot with different colors for different extreme ratios
        for ratio in sorted(df['extreme_value_ratio'].unique()):
            ratio_data = df[df['extreme_value_ratio'] == ratio]
            plt.scatter(
                ratio_data['dataset_size'], 
                ratio_data['final_weight'],
                label=f"{int(ratio*100)}% extreme",
                alpha=0.6,
                s=50
            )
        
        plt.xlabel('Dataset Size')
        plt.ylabel('Final Weight')
        plt.title('Final Weight vs Dataset Size by Extreme Value Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        scatter_path = self.output_dir / "weight_vs_size_scatter.png"
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(scatter_path))
        
        logger.info(f"Generated {len(plot_files)} summary graphs")
        return plot_files
    
    def generate_comprehensive_report(self) -> str:
        """
        Generate a comprehensive analysis report.
        
        Returns:
            Path to the generated report file
        """
        df = self._prepare_dataframe()
        
        # Generate all analyses
        weight_plots = self.create_weight_distribution_plots()
        bias_plots = self.create_bias_distribution_plots()
        summary_plots = self.create_summary_graphs()
        extreme_effects = self.compare_extreme_value_effects()
        size_trends = self.analyze_dataset_size_trends()
        
        # Create report
        report_lines = [
            "# Perceptron Extreme Values Experiment - Analysis Report",
            f"Generated from {len(df)} experimental runs",
            "",
            "## Experiment Overview",
            f"- Dataset sizes tested: {sorted(df['dataset_size'].unique())}",
            f"- Extreme value ratios tested: {[f'{r:.0%}' for r in sorted(df['extreme_value_ratio'].unique())]}",
            f"- Total experimental conditions: {len(df.groupby(['dataset_size', 'extreme_value_ratio']))}",
            f"- Repetitions per condition: {df['repetition'].max()}",
            "",
            "## Key Findings",
            "",
            "### Weight Analysis by Extreme Value Ratio",
        ]
        
        # Add weight analysis
        for ratio, stats in extreme_effects['weight_analysis'].items():
            report_lines.append(f"- {ratio:.0%} extreme values: mean weight = {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        report_lines.extend([
            "",
            "### Bias Analysis by Extreme Value Ratio",
        ])
        
        # Add bias analysis
        for ratio, stats in extreme_effects['bias_analysis'].items():
            report_lines.append(f"- {ratio:.0%} extreme values: mean bias = {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        report_lines.extend([
            "",
            "### Expected vs Actual Values",
            "- Expected weight: 2.0 (from y = 2x + 1)",
            "- Expected bias: 1.0 (from y = 2x + 1)",
            "",
            "### Convergence Analysis",
        ])
        
        # Add convergence analysis
        for ratio, stats in extreme_effects['convergence_analysis'].items():
            report_lines.append(f"- {ratio:.0%} extreme values: {stats['convergence_rate']:.1%} convergence rate")
        
        report_lines.extend([
            "",
            "### Dataset Size Effects",
        ])
        
        # Add size effects for weights
        for size, stats in extreme_effects['dataset_size_effects_weights'].items():
            report_lines.append(f"- Size {size}: mean weight = {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        report_lines.extend([
            "",
            "### Dataset Size Effects on Bias",
        ])
        
        # Add size effects for weights
        for size, stats in extreme_effects['dataset_size_effects'].items():
            report_lines.append(f"- Size {size}: mean weight = {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        report_lines.extend([
            "",
            "## Generated Visualizations",
        ])
        
        # Add plot references
        for plot_path in weight_plots + summary_plots:
            plot_name = Path(plot_path).name
            report_lines.append(f"- {plot_name}")
        
        report_lines.extend([
            "",
            "## Statistical Summary",
            f"- Overall convergence rate: {df['converged'].mean():.1%}",
            f"- Average training time: {df['execution_time'].mean():.2f} seconds",
            f"- Average training iterations: {df['training_iterations'].mean():.1f}",
            f"- Weight range: {df['final_weight'].min():.4f} to {df['final_weight'].max():.4f}",
        ])
        
        # Save report (will be moved to reports subdirectory by main script)
        report_path = self.output_dir / "experiment_analysis_report.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Generated comprehensive analysis report: {report_path}")
        return str(report_path)
    
    def export_analysis_data(self) -> str:
        """
        Export processed analysis data to CSV.
        
        Returns:
            Path to exported analysis data file
        """
        df = self._prepare_dataframe()
        
        # Add derived columns for analysis
        df['weight_deviation_from_mean'] = df['final_weight'] - df.groupby('extreme_value_ratio')['final_weight'].transform('mean')
        df['bias_deviation_from_mean'] = df['final_bias'] - df.groupby('extreme_value_ratio')['final_bias'].transform('mean')
        df['weight_deviation_from_expected'] = df['final_weight'] - 2.0  # Expected weight is 2.0
        df['bias_deviation_from_expected'] = df['final_bias'] - 1.0      # Expected bias is 1.0
        df['relative_execution_time'] = df['execution_time'] / df.groupby('dataset_size')['execution_time'].transform('mean')
        
        analysis_path = self.output_dir / "analysis_data.csv"
        df.to_csv(analysis_path, index=False)
        
        logger.info(f"Exported analysis data: {analysis_path}")
        return str(analysis_path)