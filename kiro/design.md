# Design Document

## Overview

This system implements an experiment on the correlation of extreme values in training data and resulting perceptron weight distributions. The design builds upon an existing perceptron implementation while adding rigorous statistical controls, data validation, and result analysis.

The system will execute 250 experimental runs (25 conditions × 10 repetitions) systematically varying dataset size and extreme value proportions while maintaining statistical comparability through controlled mean and standard deviation.

## Architecture

### Core Components

```
ExperimentRunner
├── DatasetGenerator     # Generates and validates datasets
│   └── StatisticalValidator # Internal dependency for dataset validation
├── PerceptronTrainer   # Handles training and uses ResultsCollector internally
│   └── ResultsCollector # Internal dependency for data collection
└── AnalysisEngine      # Generates visualizations and statistical analysis
```

### Data Flow

1. **Configuration** → ExperimentRunner loads experimental parameters
2. **Dataset Generation & Validation** → DatasetGenerator creates controlled datasets using internal StatisticalValidator
3. **Training & Collection** → PerceptronTrainer executes training and calls internal ResultsCollector
4. **Analysis** → AnalysisEngine generates CSV exports and visualizations

## Components and Interfaces

### DatasetGenerator

**Purpose**: Generate and validate datasets with controlled statistical properties and extreme value proportions.

**Key Methods**:
- `__init__(statistical_validator)` → Initialize with StatisticalValidator dependency
- `generate_validated_dataset(size, extreme_ratio, target_mean, target_std)` → (X, y, metadata)
- `calculate_iqr_threshold(data)` → threshold_value
- `_generate_initial_dataset()` → raw_dataset (private method)

**Statistical Control Strategy**:
- Generate initial dataset with target statistical properties
- Identify extreme values using 1.5 * IQR threshold
- Iteratively adjust values to achieve exact extreme value proportion while maintaining mean/std within 1% tolerance
- Use internal StatisticalValidator to ensure all datasets meet requirements before returning

### StatisticalValidator

**Purpose**: Validate dataset statistical properties (used internally by DatasetGenerator).

**Key Methods**:
- `validate_statistical_properties(dataset, target_mean, target_std)` → validation_result
- `calculate_tolerance_bounds(target_value, tolerance_pct)` → (min_bound, max_bound)
- `verify_extreme_value_proportion(data, expected_ratio)` → boolean

**Validation Criteria**:
- Mean within 1% of target value
- Standard deviation within 1% of target value
- Exact extreme value proportion (1.5 * IQR threshold)

### PerceptronTrainer

**Purpose**: Train perceptrons with consistent initialization and coordinate result collection.

**Key Methods**:
- `__init__(results_collector)` → Initialize with ResultsCollector dependency
- `train_and_record(X, y, dataset_metadata, experiment_id)` → success_status
- `create_perceptron(input_dim, seed)` → model
- `extract_training_metrics(training_history)` → metrics_dict

**Training Configuration**:
- Single linear layer (Dense(1, activation="linear"))
- SGD optimizer with consistent learning rate
- MSE loss function
- Fixed epoch count with early stopping on convergence failure
- Deterministic weight initialization using experiment-specific seeds
- Uses internal ResultsCollector to record results immediately after training

### ResultsCollector

**Purpose**: Handle data collection and aggregation (used internally by PerceptronTrainer).

**Key Methods**:
- `record_experiment_result(experiment_data)` → None
- `get_all_results()` → List[experiment_result]
- `export_raw_data_csv(filepath)` → None
- `generate_summary_statistics()` → summary_dict

**Data Schema**:
```python
{
    'experiment_id': str,
    'dataset_size': int,
    'extreme_value_ratio': float,
    'repetition': int,
    'final_weights': List[float],
    'training_iterations': int,
    'convergence_status': str,
    'dataset_mean': float,
    'dataset_std': float,
    'extreme_value_count': int,
    'training_time': float
}
```



### ExperimentRunner

**Purpose**: Orchestrate the complete experimental workflow.

**Key Methods**:
- `run_all_experiments()` → None
- `export_results()` → None (accesses results through PerceptronTrainer)
- `generate_analysis()` → None (delegates to AnalysisEngine)

### AnalysisEngine

**Purpose**: Generate visualizations and statistical comparisons.

**Key Methods**:
- `create_weight_distribution_plots()` → plot_files
- `compare_extreme_value_effects()` → statistical_comparison
- `analyze_dataset_size_trends()` → trend_analysis

**Visualization Types**:
- Weight distribution histograms by condition
- Weight evolution trends across dataset sizes
- Statistical comparison plots (box plots, violin plots)
- Convergence rate analysis by condition

## Data Models

### ExperimentConfiguration
```python
@dataclass
class ExperimentConfiguration:
    dataset_sizes: List[int] = [10, 50, 100, 500, 1000]
    extreme_value_ratios: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0]
    repetitions_per_condition: int = 10
    target_mean: float = 50.0
    target_std: float = 5.0
    statistical_tolerance: float = 0.01  # 1%
    training_epochs: int = 50
    random_seed_base: int = 42
```

### ExperimentResult
```python
@dataclass
class ExperimentResult:
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
```

### DatasetProperties
```python
@dataclass
class DatasetProperties:
    actual_mean: float
    actual_std: float
    extreme_value_count: int
    iqr_threshold: float
    statistical_validation_passed: bool
```

## Error Handling

### Training Failures
- **Non-convergence**: Record failure status, continue with remaining experiments
- **Numerical instability**: Log error details, use fallback initialization
- **Memory issues**: Implement batch processing for large datasets

### Data Generation Failures
- **Statistical constraint violations**: Retry with adjusted parameters up to 3 times
- **Extreme value generation failures**: Fall back to manual extreme value placement
- **Validation failures**: Log detailed statistics and continue with warning

### System Failures
- **File I/O errors**: Implement robust file handling with backup locations
- **Memory constraints**: Process experiments in batches if needed
- **Progress tracking**: Checkpoint system to resume interrupted experiments

## Testing Strategy

### Unit Tests
- **DatasetGenerator**: Verify statistical properties, extreme value proportions
- **StatisticalValidator**: Test tolerance calculations, validation logic
- **PerceptronTrainer**: Verify consistent initialization, weight extraction
- **ResultsCollector**: Test data aggregation, CSV export format

### Integration Tests
- **End-to-end experiment execution**: Single condition with known expected results
- **Statistical validation pipeline**: Verify complete data generation → validation flow
- **Results export**: Verify CSV format and data integrity

### Validation Tests
- **Statistical property maintenance**: Verify mean/std consistency across conditions
- **Extreme value definition**: Validate 1.5 * IQR threshold implementation
- **Reproducibility**: Verify identical results with same random seeds
- **Data completeness**: Ensure all 250 experiments produce valid results