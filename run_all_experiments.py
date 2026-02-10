"""
Main Experiment Runner
Run all MetaGB vs XGBoost comparison experiments.

Usage:
    python run_all_experiments.py --exp all      # Run all experiments
    python run_all_experiments.py --exp 1        # Run Experiment 1 only
    python run_all_experiments.py --exp 2a       # Run Experiment 2A only
    python run_all_experiments.py --exp 2b       # Run Experiment 2B only
    python run_all_experiments.py --exp 2        # Run Experiment 2A and 2B
"""

import sys
import os
import argparse
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.hyperparams import RANDOM_SEED


def run_experiment1(seed: int):
    """Run Experiment 1: Real datasets comparison."""
    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENT 1: Real Datasets Performance Comparison")
    print("=" * 80)

    from exp1_real_datasets import run_experiment1 as exp1_main
    exp1_main(seed=seed)


def run_experiment2a(seed: int):
    """Run Experiment 2A: Synthetic nonlinear robustness test."""
    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENT 2A: Synthetic Nonlinear Robustness Test")
    print("=" * 80)

    from exp2A_synthetic_nonlinear import run_experiment2a as exp2a_main
    exp2a_main(seed=seed)


def run_experiment2b(seed: int):
    """Run Experiment 2B: Feature interaction pattern test."""
    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENT 2B: Feature Interaction Pattern Test")
    print("=" * 80)

    from exp2B_synthetic_interaction import run_experiment2b as exp2b_main
    exp2b_main(seed=seed)


def main():
    parser = argparse.ArgumentParser(
        description='MetaGB vs XGBoost Comparison Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experiments:
  1   - Real datasets performance comparison (TabArena regression datasets)
  2a  - Synthetic nonlinear robustness test (square, sin, sigmoid patterns)
  2b  - Feature interaction pattern test (piecewise, conditional, sparse)
  2   - Both synthetic experiments (2a + 2b)
  all - All experiments (1 + 2a + 2b)

Examples:
  python run_all_experiments.py --exp all
  python run_all_experiments.py --exp 1 --seed 123
  python run_all_experiments.py --exp 2a 2b
        """
    )

    parser.add_argument(
        '--exp', '-e',
        nargs='+',
        choices=['1', '2a', '2b', '2', 'all'],
        default=['all'],
        help='Experiments to run (default: all)'
    )

    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=RANDOM_SEED,
        help=f'Random seed for reproducibility (default: {RANDOM_SEED})'
    )

    args = parser.parse_args()

    # Determine which experiments to run
    experiments = set()
    for exp in args.exp:
        if exp == 'all':
            experiments.update(['1', '2a', '2b'])
        elif exp == '2':
            experiments.update(['2a', '2b'])
        else:
            experiments.add(exp)

    print("=" * 80)
    print("MetaGB vs XGBoost Comparison Experiments")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Random seed: {args.seed}")
    print(f"Experiments to run: {sorted(experiments)}")
    print("=" * 80)

    # Run experiments in order
    start_time = datetime.now()

    if '1' in experiments:
        try:
            run_experiment1(args.seed)
        except Exception as e:
            print(f"ERROR in Experiment 1: {e}")
            import traceback
            traceback.print_exc()

    if '2a' in experiments:
        try:
            run_experiment2a(args.seed)
        except Exception as e:
            print(f"ERROR in Experiment 2A: {e}")
            import traceback
            traceback.print_exc()

    if '2b' in experiments:
        try:
            run_experiment2b(args.seed)
        except Exception as e:
            print(f"ERROR in Experiment 2B: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 80)
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration}")
    print("=" * 80)

    # Print output locations
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("\nOutput locations:")
    print(f"  Results: {os.path.join(base_dir, 'results')}")
    print(f"  Logs:    {os.path.join(base_dir, 'logs')}")
    print(f"  Plots:   {os.path.join(base_dir, 'plots')}")


if __name__ == "__main__":
    main()
