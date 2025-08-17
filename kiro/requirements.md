# Requirements Document

## Introduction

This experiment investigates the relationship between training data characteristics and perceptron weight distributions. Specifically, it tests whether extreme values in training data lead to different weight patterns compared to data with similar values, while maintaining comparable statistical properties across datasets. The experiment will systematically vary both dataset size (10, 50, 100, 500, 1000 data points) and the proportion of extreme values (0%, 25%, 50%, 75%, 100%) across 250 total experimental runs.

## Requirements

### Requirement 1

**User Story:** As a researcher, I want to generate datasets with controlled proportions of extreme values, so that I can study their impact on perceptron learning while maintaining statistical comparability.

#### Acceptance Criteria

1. WHEN generating a dataset THEN the system SHALL create data points with a target mean and standard deviation that serves as the baseline for comparison
2. WHEN specifying extreme value percentage THEN the system SHALL generate that exact proportion of values that are classified as extreme according to the 1.5 * IQR threshold
3. WHEN creating extreme values THEN the system SHALL define extreme values as those exceeding 1.5 * IQR (Interquartile Range) from the median of the baseline distribution
4. WHEN validating dataset properties THEN the system SHALL record the actual mean and standard deviation for each generated dataset to enable statistical comparison across conditions

### Requirement 2

**User Story:** As a researcher, I want to systematically test all defined combinations of dataset sizes and extreme value proportions with multiple repetitions, so that I can comprehensively analyze the experimental space with statistical reliability.

#### Acceptance Criteria

1. WHEN executing the full experiment THEN the system SHALL test all 25 combinations (5 dataset sizes: 10, 50, 100, 500, 1000 data points × 5 extreme value proportions: 0%, 25%, 50%, 75%, 100%)
2. WHEN running each combination THEN the system SHALL execute exactly 10 repetitions for a total of 250 runs
3. WHEN training a perceptron THEN the system SHALL initialize perceptron weights consistently across all experiments
4. WHEN training fails to converge OR any repetition fails THEN the system SHALL record the failure outcome and continue with remaining experiments
5. WHEN running repetitions THEN the system SHALL use different random seeds for each repetition
6. WHEN progressing through experiments THEN the system SHALL provide clear progress indicators

### Requirement 3

**User Story:** As a researcher, I want to collect comprehensive data about each experimental run, so that I can perform detailed analysis of perceptron behavior patterns.

#### Acceptance Criteria

1. WHEN a perceptron training completes THEN the system SHALL record final weight values, training iterations, convergence status, and training metrics
2. WHEN recording results THEN the system SHALL include dataset characteristics (size, extreme value percentage, statistical properties)
3. WHEN all experiments complete THEN the system SHALL export raw data in CSV format

### Requirement 4

**User Story:** As a researcher, I want to visualize and analyze experimental results, so that I can understand the relationship between extreme values and perceptron weights.

#### Acceptance Criteria

1. WHEN experiments complete THEN the system SHALL generate summary graphs comparing weight distributions across conditions
2. WHEN analyzing results THEN the system SHALL provide statistical comparisons between extreme value proportions and dataset sizes