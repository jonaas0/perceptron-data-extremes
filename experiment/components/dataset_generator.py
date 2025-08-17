"""
Dataset generation with integrated statistical validation.
"""

import numpy as np
from typing import Tuple
import logging
from ..models import DatasetProperties
from ..utils import calculate_iqr_bounds
from .statistical_validator import StatisticalValidator


logger = logging.getLogger(__name__)


class DatasetGenerator:
    """Generate datasets with controlled statistical properties and extreme value proportions."""
    
    def __init__(self, statistical_tolerance: float = 0.01):
        """
        Initialize dataset generator with integrated statistical validator.
        
        Args:
            statistical_tolerance: Tolerance for statistical properties (default 1%)
        """
        self.validator = StatisticalValidator(tolerance=statistical_tolerance)
    
    def _generate_dataset_base(
        self, 
        size: int, 
        extreme_ratio: float,
        target_mean: float
    ) -> np.ndarray:
        """
        Generate dataset with requested extreme ratio and adjust mean via scaling.
        
        Args:
            size: Number of points
            extreme_ratio: Proportion of extreme values (0.0 – 1.0)
            target_mean: Target mean value
            
        Returns:
            np.ndarray with values in [0, 200% of mean] and mean ≈ target_mean
        """
        # counts
        n_extreme = int(size * extreme_ratio)
        n_normal = size - n_extreme
        n_low = n_extreme // 2
        n_high = n_extreme - n_low

        m = float(target_mean)

        # ranges
        low_lo,  low_hi  = 0.0,      0.2 * m
        norm_lo, norm_hi = 0.2 * m,  1.8 * m
        high_lo, high_hi = 1.8 * m,  2.0 * m

        # draws (safe for zero counts)
        normal = np.random.uniform(norm_lo,  norm_hi,  n_normal) if n_normal else np.empty(0)
        low    = np.random.uniform(low_lo,   low_hi,   n_low)    if n_low    else np.empty(0)
        high   = np.random.uniform(high_lo,  high_hi,  n_high)   if n_high   else np.empty(0)

        data = np.concatenate([normal, low, high])
        np.random.shuffle(data)

        # scale to enforce target mean
        cur_mean = float(data.mean()) if data.size else m
        if cur_mean > 0.0:
            data *= (m / cur_mean)

        # clip hard bounds [0, 200%]
        np.clip(data, 0.0, 2.0 * m, out=data)
        return data

    def _generate_dataset_with_correct_mean(
        self, 
        size: int, 
        extreme_ratio: float,
        target_mean: float,
        max_mean_attempts: int = 20
    ) -> np.ndarray:
        """
        Try up to max_mean_attempts to generate a dataset within tolerance of target mean.
        """
        for _ in range(max_mean_attempts):
            data = self._generate_dataset_base(size, extreme_ratio, target_mean)
            actual_mean = np.mean(data)
            mean_min, mean_max = self.validator.calculate_tolerance_bounds(target_mean)
            if mean_min <= actual_mean <= mean_max:
                return data
        return None

    def generate_validated_dataset(
        self, 
        size: int, 
        extreme_ratio: float, 
        target_mean: float, 
        max_retries: int = 3
    ) -> Tuple[np.ndarray, np.ndarray, DatasetProperties]:
        """
        Generate a validated dataset with controlled statistical properties.
        """
        best_data = None
        best_properties = None
        best_validation_score = -1
        
        for attempt in range(max_retries):
            try:
                data = self._generate_dataset_with_correct_mean(size, extreme_ratio, target_mean)
                if data is None:
                    logger.warning(f"Failed to generate dataset with correct mean after attempts (retry {attempt + 1})")
                    continue
                
                # validate
                validation = self.validator.validate_statistical_properties(
                    data, target_mean, extreme_ratio
                )
                
                _, _, iqr_threshold = calculate_iqr_bounds(data)
                _, actual_extreme_ratio, extreme_count = self.validator.verify_extreme_value_proportion(
                    data, extreme_ratio
                )
                
                properties = DatasetProperties(
                    actual_mean=validation.actual_mean,
                    extreme_value_count=extreme_count,
                    iqr_threshold=iqr_threshold,
                    statistical_validation_passed=validation.passed
                )
                
                score = int(validation.mean_valid) + int(validation.extreme_ratio_valid)
                if score > best_validation_score:
                    best_validation_score = score
                    best_data = data
                    best_properties = properties
                
                if validation.passed:
                    break
                    
                logger.warning(f"Dataset generation attempt {attempt + 1} failed validation: {validation.message}")
                
            except Exception as e:
                logger.error(f"Dataset generation attempt {attempt + 1} failed with error: {e}")
                continue
        
        if best_data is None:
            raise RuntimeError(f"Failed to generate valid dataset after {max_retries} attempts")
        
        X = best_data.reshape(-1, 1)
        y = (2 * best_data + 1).reshape(-1, 1)
        
        if not best_properties.statistical_validation_passed:
            logger.warning(f"Using best available dataset (score: {best_validation_score}/2)")
        
        return X, y, best_properties