"""
Perceptron training with integrated result collection.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Tuple, Optional
import time
import logging
from ..models import ExperimentResult, ConvergenceStatus, DatasetProperties
from ..utils import generate_experiment_id
from .results_collector import ResultsCollector


logger = logging.getLogger(__name__)


class WeightTracker(keras.callbacks.Callback):
    """Callback to track weight evolution during training."""
    
    def __init__(self):
        super().__init__()
        self.weights_per_epoch = []
        self.losses_per_epoch = []
    
    def on_epoch_end(self, epoch, logs=None):
        """Record weights and loss at the end of each epoch."""
        if logs is None:
            logs = {}
        
        # Get current weights (assuming single dense layer)
        weights, bias = self.model.get_weights()
        self.weights_per_epoch.append(weights[0][0])  # First weight
        self.losses_per_epoch.append(logs.get('loss', 0.0))


class PerceptronTrainer:
    """Train perceptrons with consistent initialization and coordinate result collection."""
    
    def __init__(self, learning_rate: float = 0.1, max_epochs: int = 1000):
        """
        Initialize perceptron trainer with integrated results collector.
        
        Args:
            learning_rate: Learning rate for SGD optimizer (optimal for bias convergence)
            max_epochs: Maximum number of training epochs (increased for better convergence)
        """
        self.results_collector = ResultsCollector()
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.convergence_threshold = 1e-5  # Stricter convergence threshold
        self.patience = 25  # More patience for better convergence
    
    def create_perceptron(self, input_dim: int, seed: int) -> keras.Model:
        """
        Create a perceptron model with consistent initialization.
        
        Args:
            input_dim: Number of input features
            seed: Random seed for reproducible initialization
            
        Returns:
            Compiled Keras model
        """
        # Set seeds for reproducible initialization
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        # Create simple perceptron (single linear layer)
        model = keras.Sequential([
            keras.layers.Dense(
                1, 
                activation='linear', 
                input_shape=(input_dim,),
                kernel_initializer=keras.initializers.RandomNormal(seed=seed),
                bias_initializer=keras.initializers.Zeros()
            )
        ])
        
        # Compile with SGD optimizer and MSE loss
        optimizer = keras.optimizers.SGD(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def extract_training_metrics(self, history: keras.callbacks.History, tracker: WeightTracker) -> Dict[str, float]:
        """
        Extract training metrics from training history and weight tracker.
        
        Args:
            history: Keras training history
            tracker: Weight tracking callback
            
        Returns:
            Dictionary of training metrics
        """
        metrics = {}
        
        if history.history:
            # Final loss and metrics
            metrics['final_loss'] = float(history.history['loss'][-1])
            metrics['final_mae'] = float(history.history['mae'][-1])
            
            # Initial loss for comparison
            metrics['initial_loss'] = float(history.history['loss'][0])
            
            # Loss improvement
            metrics['loss_improvement'] = metrics['initial_loss'] - metrics['final_loss']
            
            # Weight evolution metrics
            if tracker.weights_per_epoch:
                weights = np.array(tracker.weights_per_epoch)
                metrics['initial_weight'] = float(weights[0])
                metrics['final_weight'] = float(weights[-1])
                metrics['weight_change'] = float(weights[-1] - weights[0])
                metrics['weight_std'] = float(np.std(weights))
        
        return metrics
    
    def _determine_convergence_status(
        self, 
        history: keras.callbacks.History, 
        epochs_completed: int
    ) -> ConvergenceStatus:
        """
        Determine convergence status based on training history.
        
        Args:
            history: Keras training history
            epochs_completed: Number of epochs actually completed
            
        Returns:
            ConvergenceStatus enum value
        """
        if not history.history or 'loss' not in history.history:
            return ConvergenceStatus.ERROR
        
        final_loss = history.history['loss'][-1]
        
        # Check for numerical instability
        if np.isnan(final_loss) or np.isinf(final_loss):
            return ConvergenceStatus.NUMERICAL_INSTABILITY
        
        # Check for convergence based on loss threshold
        if final_loss < self.convergence_threshold:
            return ConvergenceStatus.CONVERGED
        
        # Check if training was stopped early due to no improvement
        if epochs_completed < self.max_epochs:
            # If stopped early but loss is reasonable, consider converged
            if final_loss < 1.0:  # Reasonable loss threshold
                return ConvergenceStatus.CONVERGED
            else:
                return ConvergenceStatus.FAILED_TO_CONVERGE
        
        # Completed all epochs
        if final_loss < 1.0:  # Reasonable final loss
            return ConvergenceStatus.CONVERGED
        else:
            return ConvergenceStatus.FAILED_TO_CONVERGE
    
    def train_and_record(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        dataset_properties: DatasetProperties,
        dataset_size: int,
        extreme_value_ratio: float,
        repetition: int,
        seed: int
    ) -> bool:
        """
        Train perceptron and record results using internal ResultsCollector.
        
        Args:
            X: Input features
            y: Target labels
            dataset_properties: Properties of the dataset
            dataset_size: Size of the dataset
            extreme_value_ratio: Proportion of extreme values
            repetition: Repetition number for this condition
            seed: Random seed for this training run
            
        Returns:
            True if training completed successfully, False otherwise
        """
        experiment_id = generate_experiment_id(dataset_size, extreme_value_ratio, repetition)
        start_time = time.time()
        
        try:
            # Create and configure model
            model = self.create_perceptron(X.shape[1], seed)
            
            # Set up tracking
            weight_tracker = WeightTracker()
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=self.patience,
                restore_best_weights=True,
                verbose=0
            )
            
            # Train the model
            history = model.fit(
                X, y,
                epochs=self.max_epochs,
                verbose=0,
                callbacks=[weight_tracker, early_stopping]
            )
            
            # Extract results
            final_weights, final_bias = model.get_weights()
            training_metrics = self.extract_training_metrics(history, weight_tracker)
            convergence_status = self._determine_convergence_status(history, len(history.history['loss']))
            execution_time = time.time() - start_time
            
            # Create experiment result
            result = ExperimentResult(
                experiment_id=experiment_id,
                dataset_size=dataset_size,
                extreme_value_ratio=extreme_value_ratio,
                repetition=repetition,
                final_weights=final_weights.flatten().tolist(),
                final_bias=float(final_bias[0]),
                training_iterations=len(history.history['loss']),
                convergence_status=convergence_status,
                training_metrics=training_metrics,
                dataset_properties=dataset_properties,
                execution_time=execution_time
            )
            
            # Record result
            self.results_collector.record_experiment_result(result)
            
            logger.info(f"Training completed for {experiment_id}: {convergence_status.value}")
            return True
            
        except Exception as e:
            # Record failed experiment
            execution_time = time.time() - start_time
            
            failed_result = ExperimentResult(
                experiment_id=experiment_id,
                dataset_size=dataset_size,
                extreme_value_ratio=extreme_value_ratio,
                repetition=repetition,
                final_weights=[],
                final_bias=0.0,
                training_iterations=0,
                convergence_status=ConvergenceStatus.ERROR,
                training_metrics={'error': str(e)},
                dataset_properties=dataset_properties,
                execution_time=execution_time
            )
            
            self.results_collector.record_experiment_result(failed_result)
            logger.error(f"Training failed for {experiment_id}: {e}")
            return False
    
    def get_results_collector(self) -> ResultsCollector:
        """Get the internal results collector for accessing results."""
        return self.results_collector
    
    def export_results(self, filepath: str) -> None:
        """Export all collected results to CSV."""
        self.results_collector.export_raw_data_csv(filepath)
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics of all training results."""
        return self.results_collector.generate_summary_statistics()
    
    def clear_results(self) -> None:
        """Clear all collected results."""
        self.results_collector.clear_results()