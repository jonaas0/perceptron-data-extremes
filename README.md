# Perceptron Data Extremes
This repository contains a small experiment on the relation of data and weights. This is done by varying the amount of extreme data in the Dataset and analyzing the results. A substantial part of the code was derived from Claude Sonnet 4.0. 

## Experiment Design

- **25 experimental conditions**: 5 dataset sizes × 5 extreme value proportions
- **250 total runs**: 10 repetitions per condition
- **Dataset sizes**: 10, 50, 100, 500, 1000 data points
- **Extreme value proportions**: 0%, 25%, 50%, 75%, 100%
- **Extreme value definition**: Values smaller then 20% or bigger then 180% of the mean 

Under outputs theres a folder containing the results of the current iteration of the experiment. Depending on your machine running this yourself may take 5-30 Minutes. If you plan on running the experiment, you may create a virtual environment for the necessary dependencies. They are listed in the requirements.txt.

### Main Experiment
```bash
# Run quick test (8 experiments) - creates outputs/test_experiment_YYYYMMDD_HHMMSS/
python main_experiment.py --mode test

# Run full experiment (250 experiments) - creates outputs/full_experiment_YYYYMMDD_HHMMSS/
python main_experiment.py --mode full