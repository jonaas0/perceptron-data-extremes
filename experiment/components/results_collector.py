"""
Results collection and aggregation for experimental data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging
from pathlib import Path
from ..models import ExperimentResult, ConvergenceStatus


logger = logging.getLogger(__name__)


class ResultsCollector:
    """Handle data collection and aggregation for experimental results."""
    
    def __init__(self):
        """Initialize results collector with empty storage."""
        self.results: List[ExperimentResult] = []
    
    def record_experiment_result(self, result: ExperimentResult) -> None:
        """
        Record a single experimental result.
        
        Args:
            result: ExperimentResult object containing all experimental data
        """
        self.results.append(result)
        logger.debug(f"Recorded result for experiment {result.experiment_id}")
    
    def get_all_results(self) -> List[ExperimentResult]:
        """
        Get all recorded experimental results.
        
        Returns:
            List of all ExperimentResult objects
        """
        return self.results.copy()
    
    def get_results_count(self) -> int:
        """Get the total number of recorded results."""
        return len(self.results)
    
    def get_results_by_condition(self, dataset_size: int, extreme_ratio: float) -> List[ExperimentResult]:
        """
        Get results for a specific experimental condition.
        
        Args:
            dataset_size: Dataset size to filter by
            extreme_ratio: Extreme value ratio to filter by
            
        Returns:
            List of results matching the condition
        """
        return [
            result for result in self.results
            if result.dataset_size == dataset_size and result.extreme_value_ratio == extreme_ratio
        ]
    
    def export_raw_data_csv(self, filepath: str) -> None:
        """
        Export all experimental results to CSV format.
        
        Args:
            filepath: Path where to save the CSV file
        """
        if not self.results:
            logger.warning("No results to export")
            return
        
        # Convert results to flat dictionary format for CSV
        rows = []
        for result in self.results:
            # Base row data
            row = {
                'experiment_id': result.experiment_id,
                'dataset_size': result.dataset_size,
                'extreme_value_ratio': result.extreme_value_ratio,
                'repetition': result.repetition,
                'final_bias': result.final_bias,
                'training_iterations': result.training_iterations,
                'convergence_status': result.convergence_status.value,
                'execution_time': result.execution_time,
                'dataset_mean': result.dataset_properties.actual_mean,
                'extreme_value_count': result.dataset_properties.extreme_value_count,
                'iqr_threshold': result.dataset_properties.iqr_threshold,
                'statistical_validation_passed': result.dataset_properties.statistical_validation_passed
            }
            
            # Add final weights (assuming single weight for simple perceptron)
            if result.final_weights:
                for i, weight in enumerate(result.final_weights):
                    row[f'final_weight_{i}'] = weight
            
            # Add training metrics
            for metric_name, metric_value in result.training_metrics.items():
                row[f'training_{metric_name}'] = metric_value
            
            rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        logger.info(f"Exported {len(rows)} results to {filepath}")
    
    def generate_summary_statistics(self) -> Dict[str, Any]:
        """
        Generate summary statistics across all experimental results.
        
        Returns:
            Dictionary containing summary statistics
        """
        if not self.results:
            return {"message": "No results available for summary"}
        
        # Convert to DataFrame for easier analysis
        data = []
        for result in self.results:
            data.append({
                'dataset_size': result.dataset_size,
                'extreme_value_ratio': result.extreme_value_ratio,
                'repetition': result.repetition,
                'final_weight': result.final_weights[0] if result.final_weights else None,
                'final_bias': result.final_bias,
                'training_iterations': result.training_iterations,
                'convergence_status': result.convergence_status.value,
                'execution_time': result.execution_time,
                'converged': result.convergence_status == ConvergenceStatus.CONVERGED
            })
        
        df = pd.DataFrame(data)
        
        # Calculate summary statistics
        summary = {
            'total_experiments': len(self.results),
            'unique_conditions': len(df.groupby(['dataset_size', 'extreme_value_ratio'])),
            'convergence_rate': df['converged'].mean(),
            'average_training_iterations': df['training_iterations'].mean(),
            'average_execution_time': df['execution_time'].mean(),
            'weight_statistics': {
                'mean': df['final_weight'].mean() if 'final_weight' in df.columns else None,
                'min': df['final_weight'].min() if 'final_weight' in df.columns else None,
                'max': df['final_weight'].max() if 'final_weight' in df.columns else None
            },
            'by_dataset_size': {},
            'by_extreme_ratio': {}
        }
        
        # Statistics by dataset size
        for size in df['dataset_size'].unique():
            size_data = df[df['dataset_size'] == size]
            summary['by_dataset_size'][int(size)] = {
                'count': len(size_data),
                'convergence_rate': size_data['converged'].mean(),
                'avg_weight': size_data['final_weight'].mean() if 'final_weight' in size_data.columns else None,
                'avg_iterations': size_data['training_iterations'].mean()
            }
        
        # Statistics by extreme ratio
        for ratio in df['extreme_value_ratio'].unique():
            ratio_data = df[df['extreme_value_ratio'] == ratio]
            summary['by_extreme_ratio'][float(ratio)] = {
                'count': len(ratio_data),
                'convergence_rate': ratio_data['converged'].mean(),
                'avg_weight': ratio_data['final_weight'].mean() if 'final_weight' in ratio_data.columns else None,
                'avg_iterations': ratio_data['training_iterations'].mean()
            }
        
        return summary
    
    def clear_results(self) -> None:
        """Clear all stored results."""
        self.results.clear()
        logger.info("Cleared all stored results")
    
    def get_convergence_summary(self) -> Dict[str, int]:
        """Get summary of convergence statuses."""
        status_counts = {}
        for result in self.results:
            status = result.convergence_status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        return status_counts