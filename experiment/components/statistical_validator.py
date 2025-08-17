"""
Statistical validation for dataset properties.
"""

import numpy as np
from typing import Tuple, NamedTuple
from ..utils import calculate_iqr_bounds


class ValidationResult(NamedTuple):
    """Result of statistical validation."""
    passed: bool
    mean_valid: bool
    extreme_ratio_valid: bool
    actual_mean: float
    actual_extreme_ratio: float
    message: str


class StatisticalValidator:
    """Validates dataset statistical properties against target values."""
    
    def __init__(self, tolerance: float = 0.01):
        """
        Initialize validator with tolerance percentage.
        
        Args:
            tolerance: Tolerance as decimal (0.01 = 1%)
        """
        self.tolerance = tolerance
    
    def calculate_tolerance_bounds(self, target_value: float) -> Tuple[float, float]:
        """
        Calculate tolerance bounds for a target value.
        
        Args:
            target_value: The target value to calculate bounds for
            
        Returns:
            Tuple of (min_bound, max_bound)
        """
        margin = abs(target_value * self.tolerance)
        return target_value - margin, target_value + margin
    
    def verify_extreme_value_proportion(
        self, 
        data: np.ndarray, 
        expected_ratio: float,
        target_mean: float = 10.0
    ) -> Tuple[bool, float, int]:
        """
        Verify that the dataset has the expected proportion of extreme values.
        
        PERCENTAGE BOUNDARIES: Extreme values are ≤20% of mean OR ≥180% of mean.
        For target_mean=10.0: extreme if value ≤ 2.0 OR value ≥ 18.0
        Normal values are in range (2.0, 18.0)
        
        Args:
            data: Dataset to analyze
            expected_ratio: Expected proportion of extreme values (0.0 to 1.0)
            target_mean: Target mean value (default 10.0)
            
        Returns:
            Tuple of (is_valid, actual_ratio, extreme_count)
        """
        if len(data) == 0:
            return False, 0.0, 0
        
        # FIXED BOUNDARIES: Extreme if < 20% of mean OR > 180% of mean
        low_extreme_max = 0.2 * target_mean   # 2.0 for mean=10
        high_extreme_min = 1.8 * target_mean  # 18.0 for mean=10
        
        # Extreme if value < 2.0 OR value > 18.0 (exclusive bounds)
        extreme_mask = (data < low_extreme_max) | (data > high_extreme_min)
        extreme_count = np.sum(extreme_mask)
        
        actual_ratio = extreme_count / len(data)
        
        # Allow reasonable tolerance for extreme value proportion
        ratio_tolerance = max(0.05, 3.0 / len(data))  # 5% tolerance or 3 data points, whichever is larger
        is_valid = abs(actual_ratio - expected_ratio) <= ratio_tolerance
        
        return is_valid, actual_ratio, extreme_count
    
    def validate_statistical_properties(
        self, 
        data: np.ndarray, 
        target_mean: float, 
        expected_extreme_ratio: float = None
    ) -> ValidationResult:
        """
        Validate dataset against target statistical properties.
        
        Args:
            data: Dataset to validate
            target_mean: Target mean value
            expected_extreme_ratio: Expected proportion of extreme values (optional)
            
        Returns:
            ValidationResult with detailed validation information
        """
        if len(data) == 0:
            return ValidationResult(
                passed=False,
                mean_valid=False,
                extreme_ratio_valid=False,
                actual_mean=0.0,
                actual_extreme_ratio=0.0,
                message="Empty dataset"
            )
        
        # Calculate actual statistics
        actual_mean = np.mean(data)
        
        # Validate mean
        mean_min, mean_max = self.calculate_tolerance_bounds(target_mean)
        mean_valid = mean_min <= actual_mean <= mean_max
        
        # Validate extreme value proportion if specified
        extreme_ratio_valid = True
        actual_extreme_ratio = 0.0
        if expected_extreme_ratio is not None:
            extreme_ratio_valid, actual_extreme_ratio, _ = self.verify_extreme_value_proportion(
                data, expected_extreme_ratio, target_mean
            )
        
        # Overall validation
        passed = mean_valid and extreme_ratio_valid
        
        # Create message
        messages = []
        if not mean_valid:
            messages.append(f"Mean {actual_mean:.3f} outside bounds [{mean_min:.3f}, {mean_max:.3f}]")
        if not extreme_ratio_valid:
            messages.append(f"Extreme ratio {actual_extreme_ratio:.3f} != expected {expected_extreme_ratio:.3f}")
        
        message = "; ".join(messages) if messages else "All validations passed"
        
        return ValidationResult(
            passed=passed,
            mean_valid=mean_valid,
            extreme_ratio_valid=extreme_ratio_valid,
            actual_mean=actual_mean,
            actual_extreme_ratio=actual_extreme_ratio,
            message=message
        )