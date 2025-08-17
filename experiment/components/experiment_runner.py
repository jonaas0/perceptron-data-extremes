"""
Experiment orchestration and coordination.
"""

import logging
import time
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import numpy as np

from ..models import ExperimentConfiguration, ExperimentResult
from ..utils import set_random_seeds, generate_experiment_id
from .dataset_generator import DatasetGenerator
from .perceptron_trainer import PerceptronTrainer


logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrate the complete experimental workflow."""
    
    def __init__(self, config: Optional[ExperimentConfiguration] = None):
        """
        Initialize experiment runner with configuration.
        
        Args:
            config: Experiment configuration. If None, uses default configuration.
        """
        self.config = config or ExperimentConfiguration()
        
        # Initialize components
        self.dataset_generator = DatasetGenerator(
            statistical_tolerance=self.config.statistical_tolerance
        )
        self.perceptron_trainer = PerceptronTrainer(
            learning_rate=0.01,  # Improved learning rate for better bias convergence
            max_epochs=self.config.training_epochs
        )
        
        # Progress tracking
        self.total_experiments = (
            len(self.config.dataset_sizes) * 
            len(self.config.extreme_value_ratios) * 
            self.config.repetitions_per_condition
        )
        self.completed_experiments = 0
        self.failed_experiments = 0
        
        # Timing
        self.start_time = None
        self.end_time = None
        
        logger.info(f"ExperimentRunner initialized for {self.total_experiments} total experiments")
    
    def _generate_experiment_conditions(self) -> List[Dict]:
        """
        Generate all experimental conditions to run.
        
        Returns:
            List of experiment condition dictionaries
        """
        conditions = []
        
        for dataset_size in self.config.dataset_sizes:
            for extreme_ratio in self.config.extreme_value_ratios:
                for repetition in range(1, self.config.repetitions_per_condition + 1):
                    conditions.append({
                        'dataset_size': dataset_size,
                        'extreme_ratio': extreme_ratio,
                        'repetition': repetition,
                        'experiment_id': generate_experiment_id(dataset_size, extreme_ratio, repetition)
                    })
        
        return conditions
    
    def _run_single_experiment(self, condition: Dict) -> bool:
        """
        Run a single experimental condition.
        
        Args:
            condition: Dictionary containing experiment parameters
            
        Returns:
            True if experiment completed successfully, False otherwise
        """
        dataset_size = condition['dataset_size']
        extreme_ratio = condition['extreme_ratio']
        repetition = condition['repetition']
        experiment_id = condition['experiment_id']
        
        try:
            # Set random seed for reproducibility
            seed = self.config.random_seed_base + (repetition - 1) * 1000 + dataset_size + int(extreme_ratio * 100)
            set_random_seeds(seed)
            
            logger.debug(f"Starting experiment {experiment_id} with seed {seed}")
            
            # Generate dataset
            X, y, dataset_properties = self.dataset_generator.generate_validated_dataset(
                size=dataset_size,
                extreme_ratio=extreme_ratio,
                target_mean=self.config.target_mean,
                max_retries=3
            )
            
            # Train perceptron and record results
            success = self.perceptron_trainer.train_and_record(
                X=X,
                y=y,
                dataset_properties=dataset_properties,
                dataset_size=dataset_size,
                extreme_value_ratio=extreme_ratio,
                repetition=repetition,
                seed=seed
            )
            
            if success:
                logger.debug(f"Experiment {experiment_id} completed successfully")
                return True
            else:
                logger.warning(f"Experiment {experiment_id} failed during training")
                self.failed_experiments += 1
                return False
                
        except Exception as e:
            logger.error(f"Experiment {experiment_id} failed with exception: {e}")
            self.failed_experiments += 1
            return False
    
    def run_all_experiments(self, show_progress: bool = True) -> Dict:
        """
        Run all experimental conditions systematically.
        
        Args:
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with experiment summary
        """
        logger.info("Starting complete experimental run")
        self.start_time = time.time()
        
        # Generate all conditions
        conditions = self._generate_experiment_conditions()
        
        # Progress tracking
        if show_progress:
            progress_bar = tqdm(
                conditions, 
                desc="Running experiments",
                unit="exp",
                ncols=100
            )
        else:
            progress_bar = conditions
        
        successful_experiments = 0
        
        # Run all experiments
        for condition in progress_bar:
            success = self._run_single_experiment(condition)
            
            if success:
                successful_experiments += 1
            
            self.completed_experiments += 1
            
            # Update progress bar description
            if show_progress:
                progress_bar.set_postfix({
                    'Success': f"{successful_experiments}/{self.completed_experiments}",
                    'Failed': self.failed_experiments
                })
        
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        
        # Generate summary
        summary = {
            'total_experiments': self.total_experiments,
            'completed_experiments': self.completed_experiments,
            'successful_experiments': successful_experiments,
            'failed_experiments': self.failed_experiments,
            'success_rate': successful_experiments / self.completed_experiments if self.completed_experiments > 0 else 0,
            'total_time_seconds': total_time,
            'average_time_per_experiment': total_time / self.completed_experiments if self.completed_experiments > 0 else 0,
            'conditions_tested': {
                'dataset_sizes': self.config.dataset_sizes,
                'extreme_value_ratios': self.config.extreme_value_ratios,
                'repetitions_per_condition': self.config.repetitions_per_condition
            }
        }
        
        logger.info(f"Experimental run completed: {successful_experiments}/{self.total_experiments} successful")
        logger.info(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        
        return summary
    
    def get_results_collector(self):
        """Get the results collector from the perceptron trainer."""
        return self.perceptron_trainer.get_results_collector()
    
    def export_results(self, filepath: str) -> None:
        """
        Export all experimental results to CSV.
        
        Args:
            filepath: Path where to save the CSV file
        """
        self.perceptron_trainer.export_results(filepath)
        logger.info(f"Results exported to {filepath}")
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics of all experimental results."""
        return self.perceptron_trainer.get_summary_statistics()
    
    def clear_results(self) -> None:
        """Clear all collected results."""
        self.perceptron_trainer.clear_results()
        self.completed_experiments = 0
        self.failed_experiments = 0
        logger.info("All results cleared")
    
    def get_experiment_progress(self) -> Dict:
        """
        Get current experiment progress.
        
        Returns:
            Dictionary with progress information
        """
        progress_percentage = (self.completed_experiments / self.total_experiments * 100) if self.total_experiments > 0 else 0
        
        return {
            'completed': self.completed_experiments,
            'total': self.total_experiments,
            'failed': self.failed_experiments,
            'progress_percentage': progress_percentage,
            'is_running': self.start_time is not None and self.end_time is None
        }
    
    def run_subset_experiment(
        self, 
        dataset_sizes: Optional[List[int]] = None,
        extreme_ratios: Optional[List[float]] = None,
        repetitions: Optional[int] = None,
        show_progress: bool = True
    ) -> Dict:
        """
        Run a subset of experiments for testing or partial runs.
        
        Args:
            dataset_sizes: Subset of dataset sizes to test
            extreme_ratios: Subset of extreme ratios to test
            repetitions: Number of repetitions (overrides config)
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with experiment summary
        """
        # Create temporary config for subset
        original_config = self.config
        
        subset_config = ExperimentConfiguration(
            dataset_sizes=dataset_sizes or [self.config.dataset_sizes[0]],
            extreme_value_ratios=extreme_ratios or [self.config.extreme_value_ratios[0]],
            repetitions_per_condition=repetitions or 1,
            target_mean=self.config.target_mean,
            statistical_tolerance=self.config.statistical_tolerance,
            training_epochs=self.config.training_epochs,
            random_seed_base=self.config.random_seed_base
        )
        
        # Temporarily replace config
        self.config = subset_config
        self.total_experiments = (
            len(self.config.dataset_sizes) * 
            len(self.config.extreme_value_ratios) * 
            self.config.repetitions_per_condition
        )
        
        # Run experiments
        summary = self.run_all_experiments(show_progress)
        
        # Restore original config
        self.config = original_config
        self.total_experiments = (
            len(self.config.dataset_sizes) * 
            len(self.config.extreme_value_ratios) * 
            self.config.repetitions_per_condition
        )
        
        return summary