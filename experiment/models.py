"""
Core data models for the perceptron extreme values experiment.
"""

from dataclasses import dataclass
from typing import List, Dict
from enum import Enum


class ConvergenceStatus(Enum):
    """Enumeration for training convergence status."""
    CONVERGED = "converged"
    FAILED_TO_CONVERGE = "failed_to_converge"
    NUMERICAL_INSTABILITY = "numerical_instability"
    ERROR = "error"


@dataclass
class ExperimentConfiguration:
    """Configuration parameters for the complete experiment."""
    dataset_sizes: List[int] = None
    extreme_value_ratios: List[float] = None
    repetitions_per_condition: int = 10
    target_mean: float = 10.0
    statistical_tolerance: float = 0.01 
    training_epochs: int = 50
    random_seed_base: int = 42
    
    def __post_init__(self):
        if self.dataset_sizes is None:
            self.dataset_sizes = [10, 50, 100, 500, 1000]
        if self.extreme_value_ratios is None:
            self.extreme_value_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]


@dataclass
class DatasetProperties:
    """Properties of a generated dataset for validation and analysis."""
    actual_mean: float
    extreme_value_count: int
    iqr_threshold: float
    statistical_validation_passed: bool


@dataclass
class ExperimentResult:
    """Complete result data for a single experimental run."""
    experiment_id: str
    dataset_size: int
    extreme_value_ratio: float
    repetition: int
    final_weights: List[float]
    final_bias: float
    training_iterations: int
    convergence_status: ConvergenceStatus
    training_metrics: Dict[str, float]
    dataset_properties: DatasetProperties
    execution_time: float