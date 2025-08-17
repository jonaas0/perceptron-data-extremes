#!/usr/bin/env python3
"""
Main script to run the complete perceptron extreme values experiment.

This script orchestrates the full experimental study investigating how extreme values
in training data affect perceptron weight distributions.
"""

import argparse
import logging
import sys
from pathlib import Path
import time
from datetime import datetime

from experiment.models import ExperimentConfiguration
from experiment.components import ExperimentRunner, AnalysisEngine


def setup_logging(log_level: str = "INFO", output_dir: Path = None) -> None:
    """Set up logging configuration."""
    log_file = output_dir / 'experiment.log' if output_dir else 'outputs/experiment.log'
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )


def create_experiment_config(args) -> ExperimentConfiguration:
    """Create experiment configuration from command line arguments."""
    config = ExperimentConfiguration()
    
    # Override defaults with command line arguments if provided
    if args.dataset_sizes:
        config.dataset_sizes = args.dataset_sizes
    if args.extreme_ratios:
        config.extreme_value_ratios = args.extreme_ratios
    if args.repetitions:
        config.repetitions_per_condition = args.repetitions
    if args.epochs:
        config.training_epochs = args.epochs
    if args.target_mean:
        config.target_mean = args.target_mean
    if args.tolerance:
        config.statistical_tolerance = args.tolerance
    if args.seed:
        config.random_seed_base = args.seed
    
    return config


def create_experiment_output_directory(mode: str) -> Path:
    """Create a structured output directory for the experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if mode == 'full':
        dir_name = f"full_experiment_{timestamp}"
    else:
        dir_name = f"test_experiment_{timestamp}"
    
    output_dir = Path("outputs") / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for organization
    (output_dir / "data").mkdir(exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    (output_dir / "reports").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    
    return output_dir


def run_full_experiment(config: ExperimentConfiguration, output_dir: Path, show_progress: bool = True) -> None:
    """Run the complete experimental study."""
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Perceptron Extreme Values Experiment")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Configuration: {len(config.dataset_sizes)} sizes × {len(config.extreme_value_ratios)} ratios × {config.repetitions_per_condition} repetitions = {len(config.dataset_sizes) * len(config.extreme_value_ratios) * config.repetitions_per_condition} total experiments")
    
    # Initialize experiment runner
    runner = ExperimentRunner(config)
    
    # Run all experiments
    start_time = time.time()
    summary = runner.run_all_experiments(show_progress=show_progress)
    end_time = time.time()
    
    # Log summary
    logger.info("Experiment completed!")
    logger.info(f"Total time: {end_time - start_time:.2f} seconds ({(end_time - start_time)/60:.1f} minutes)")
    logger.info(f"Success rate: {summary['success_rate']:.1%}")
    logger.info(f"Failed experiments: {summary['failed_experiments']}")
    
    # Export raw results to data subdirectory
    results_path = output_dir / "data" / "experiment_results.csv"
    runner.export_results(str(results_path))
    logger.info(f"Raw results exported to: {results_path}")
    
    # Generate analysis with plots subdirectory
    logger.info("Generating analysis and visualizations...")
    analysis_engine = AnalysisEngine(runner.get_results_collector(), str(output_dir / "plots"))
    
    # Generate comprehensive report in reports subdirectory
    report_path = analysis_engine.generate_comprehensive_report()
    # Move report to reports subdirectory
    report_file = Path(report_path)
    new_report_path = output_dir / "reports" / report_file.name
    report_file.rename(new_report_path)
    logger.info(f"Analysis report generated: {new_report_path}")
    
    # Export analysis data to data subdirectory
    analysis_data_path = analysis_engine.export_analysis_data()
    # Move analysis data to data subdirectory
    analysis_file = Path(analysis_data_path)
    new_analysis_path = output_dir / "data" / analysis_file.name
    analysis_file.rename(new_analysis_path)
    logger.info(f"Analysis data exported: {new_analysis_path}")
    
    # Create experiment summary file
    summary_path = output_dir / "experiment_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Perceptron Extreme Values Experiment Summary\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total experiments: {summary['total_experiments']}\n")
        f.write(f"Successful experiments: {summary['successful_experiments']}\n")
        f.write(f"Failed experiments: {summary['failed_experiments']}\n")
        f.write(f"Success rate: {summary['success_rate']:.1%}\n")
        f.write(f"Total time: {end_time - start_time:.2f} seconds ({(end_time - start_time)/60:.1f} minutes)\n")
        f.write(f"Average time per experiment: {summary['average_time_per_experiment']:.2f} seconds\n")
        f.write(f"\nConfiguration:\n")
        f.write(f"Dataset sizes: {config.dataset_sizes}\n")
        f.write(f"Extreme value ratios: {config.extreme_value_ratios}\n")
        f.write(f"Repetitions per condition: {config.repetitions_per_condition}\n")
        f.write(f"Target mean: {config.target_mean}\n")
        f.write(f"Training epochs: {config.training_epochs}\n")
        f.write(f"Statistical tolerance: {config.statistical_tolerance}\n")
    
    logger.info(f"Experiment summary saved to: {summary_path}")
    logger.info("Experiment and analysis completed successfully!")


def run_quick_test(output_dir: Path, show_progress: bool = True) -> None:
    """Run a quick test with minimal configuration."""
    logger = logging.getLogger(__name__)
    
    logger.info("Running quick test experiment...")
    
    # Create minimal configuration for testing
    test_config = ExperimentConfiguration(
        dataset_sizes=[10, 50],
        extreme_value_ratios=[0.0, 0.25],
        repetitions_per_condition=5,
        target_mean=5.0,
        training_epochs=15,
        statistical_tolerance=0.05
    )
    
    run_full_experiment(test_config, output_dir, show_progress)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run perceptron extreme values experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Experiment mode
    parser.add_argument(
        '--mode', 
        choices=['full', 'test'], 
        default='test',
        help='Experiment mode: full (250 experiments) or test (8 experiments)'
    )
    
    # Configuration options
    parser.add_argument(
        '--dataset-sizes', 
        type=int, 
        nargs='+',
        help='Dataset sizes to test (default: [10, 50, 100, 500, 1000] for full mode)'
    )
    parser.add_argument(
        '--extreme-ratios', 
        type=float, 
        nargs='+',
        help='Extreme value ratios to test (default: [0.0, 0.25, 0.5, 0.75, 1.0] for full mode)'
    )
    parser.add_argument(
        '--repetitions', 
        type=int,
        help='Number of repetitions per condition (default: 10 for full mode, 2 for test mode)'
    )
    parser.add_argument(
        '--epochs', 
        type=int,
        default=10,
        help='Training epochs (default: 15)'
    )
    parser.add_argument(
        '--target-mean', 
        type=float,
        default=10.0,
        help='Target mean for datasets (default: 5.0 for full mode, 5.0 for test mode)'
    )
    parser.add_argument(
        '--tolerance', 
        type=float,
        help='Statistical tolerance (default: 0.05 for full mode, 0.05 for test mode)'
    )
    parser.add_argument(
        '--seed', 
        type=int,
        help='Random seed base (default: 42)'
    )
    
    # Output options
    parser.add_argument(
        '--no-progress', 
        action='store_true',
        help='Disable progress bar'
    )
    parser.add_argument(
        '--log-level', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Ensure outputs directory exists
    Path('outputs').mkdir(exist_ok=True)
    
    # Create structured output directory for this experiment
    output_dir = create_experiment_output_directory(args.mode)
    
    # Set up logging with experiment-specific log file
    setup_logging(args.log_level, output_dir / "logs")
    
    logger = logging.getLogger(__name__)
    logger.info(f"Created experiment directory: {output_dir}")
    
    try:
        if args.mode == 'full':
            # Run full experiment
            #config = create_experiment_config(args)
            config = ExperimentConfiguration(
                dataset_sizes=[10, 50, 100, 500, 1000],
                extreme_value_ratios=[0.0, 0.25, 0.5, 0.75, 1.0],
                repetitions_per_condition=10,
                target_mean=5.0,
                training_epochs=50,
                statistical_tolerance=0.01
            )


            run_full_experiment(config, output_dir, show_progress=not args.no_progress)
        else:
            # Run quick test
            run_quick_test(output_dir, show_progress=not args.no_progress)
            
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()