# Perceptron Extreme Values Experiment - Analysis Report
Generated from 250 experimental runs

## Experiment Overview
- Dataset sizes tested: [np.int64(10), np.int64(50), np.int64(100), np.int64(500), np.int64(1000)]
- Extreme value ratios tested: ['0%', '25%', '50%', '75%', '100%']
- Total experimental conditions: 25
- Repetitions per condition: 10

## Key Findings

### Weight Analysis by Extreme Value Ratio
- 0% extreme values: mean weight = 2.0465 ± 0.0385
- 25% extreme values: mean weight = 2.0364 ± 0.0334
- 50% extreme values: mean weight = 2.0300 ± 0.0307
- 75% extreme values: mean weight = 2.0247 ± 0.0255
- 100% extreme values: mean weight = 2.0210 ± 0.0217

### Bias Analysis by Extreme Value Ratio
- 0% extreme values: mean bias = 0.7197 ± 0.2246
- 25% extreme values: mean bias = 0.7607 ± 0.2164
- 50% extreme values: mean bias = 0.7834 ± 0.2106
- 75% extreme values: mean bias = 0.8032 ± 0.2016
- 100% extreme values: mean bias = 0.8121 ± 0.1937

### Expected vs Actual Values
- Expected weight: 2.0 (from y = 2x + 1)
- Expected bias: 1.0 (from y = 2x + 1)

### Convergence Analysis
- 0% extreme values: 100.0% convergence rate
- 25% extreme values: 100.0% convergence rate
- 50% extreme values: 100.0% convergence rate
- 75% extreme values: 100.0% convergence rate
- 100% extreme values: 100.0% convergence rate

### Dataset Size Effects
- Size 10: mean weight = 2.0758 ± 0.0157
- Size 50: mean weight = 2.0524 ± 0.0150
- Size 100: mean weight = 2.0285 ± 0.0155
- Size 500: mean weight = 2.0018 ± 0.0026
- Size 1000: mean weight = 2.0001 ± 0.0002

### Dataset Size Effects on Bias
- Size 10: mean weight = 2.0758 ± 0.0157
- Size 50: mean weight = 2.0524 ± 0.0150
- Size 100: mean weight = 2.0285 ± 0.0155
- Size 500: mean weight = 2.0018 ± 0.0026
- Size 1000: mean weight = 2.0001 ± 0.0002

## Generated Visualizations
- weight_distributions_by_condition.png
- weight_heatmap.png
- experiment_summary_graphs.png
- weight_vs_size_scatter.png

## Statistical Summary
- Overall convergence rate: 100.0%
- Average training time: 3.69 seconds
- Average training iterations: 50.0
- Weight range: 2.0000 to 2.1098