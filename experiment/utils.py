"""
Utility functions for the perceptron extreme values experiment.
"""

import numpy as np
from typing import Tuple


def generate_experiment_id(dataset_size: int, extreme_ratio: float, repetition: int) -> str:
    """Generate a unique experiment identifier."""
    return f"exp_{dataset_size}_{int(extreme_ratio*100):02d}_{repetition:02d}"


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def calculate_iqr_bounds(data: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate IQR bounds for extreme value detection.
    
    Returns:
        Tuple of (Q1, Q3, IQR_threshold) where IQR_threshold = 1.5 * IQR
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    iqr_threshold = 1.5 * iqr
    return q1, q3, iqr_threshold