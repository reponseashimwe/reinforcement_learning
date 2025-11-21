"""
Results Aggregation Script

This script aggregates results from all trained models and generates
comparison tables and plots for the final report.

Usage:
    python evaluation/aggregate_results.py --logs_dir logs --output_dir evaluation
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import glob


def load_csv_results(algorithm: str, logs_dir: str) -> pd.DataFrame:
    """
    Load CSV results for a given algorithm.

    Args:
        algorithm: Algorithm name ('dqn', 'ppo', 'a2c', 'reinforce')
        logs_dir: Directory containing log files

    Returns:
        DataFrame with results
    """
    results_path = os.path.join(logs_dir, algorithm, "full_results.csv")

    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        df['algorithm'] = algorithm.upper()
        return df
    else:
        print(f"Warning: Results file not found: {results_path}")
        return pd.DataFrame()


def find_best_config(df: pd.DataFrame) -> Dict:
    """
    Find the best configuration based on mean reward.

    Args:
        df: DataFrame with results

    Returns:
        Dictionary with best config information
    """
    if df.empty:
        return {}

    # Group by config_id and compute statistics
    grouped = df.groupby('config_id').agg({
        'mean_reward': ['mean', 'std'],
        'triage_accuracy': ['mean', 'std'],
        'avg_wait_time': ['mean', 'std'],
        'steps_per_episode': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    grouped.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                       for col in grouped.columns.values]

    # Find best by mean reward
    best_idx = grouped['mean_reward_mean'].idxmax()
    best_row = grouped.iloc[best_idx]

    return {
        'config_id': best_row['config_id'],
        'mean_reward': best_row['mean_reward_mean'],
        'std_reward': best_row['mean_reward_std'],
        'triage_accuracy': best_row['triage_accuracy_mean'],
        'triage_accuracy_std': best_row['triage_accuracy_std'],
        'avg_wait_time': best_row['avg_wait_time_mean'],
        'steps_per_episode': best_row['steps_per_episode_mean']
    }


def create_comparison_table(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create comparison table of best configs across algorithms.

    Args:
        results: Dictionary mapping algorithm names to DataFrames

    Returns:
        Comparison DataFrame
    """
    comparison_data = []

    for algorithm, df in results.items():
        if df.empty:
            continue

        best = find_best_config(df)
        if best:
            comparison_data.append({
                'Algorithm': algorithm.upper(),
                'Config': best['config_id'],
                'Mean Reward': f"{best['mean_reward']:.2f} ± {best['std_reward']:.2f}",
                'Triage Accuracy (%)': f"{best['triage_accuracy']:.1f} ± {best['triage_accuracy_std']:.1f}",
                'Avg Wait Time': f"{best['avg_wait_time']:.2f}",
                'Steps/Episode': f"{best['steps_per_episode']:.1f}"
            })

    return pd.DataFrame(comparison_data)


def plot_learning_curves(results: Dict[str, pd.DataFrame], output_dir: str):
    """
    Plot learning curves for all algorithms.

    Args:
        results: Dictionary mapping algorithm names to DataFrames
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Learning Curves Comparison', fontsize=16, fontweight='bold')

    algorithms = ['dqn', 'ppo', 'a2c', 'reinforce']
    colors = {'dqn': '#1f77b4', 'ppo': '#ff7f0e', 'a2c': '#2ca02c', 'reinforce': '#d62728'}

    for idx, algorithm in enumerate(algorithms):
        ax = axes[idx // 2, idx % 2]
        df = results.get(algorithm, pd.DataFrame())

        if not df.empty:
            # Plot training curve for best config
            best = find_best_config(df)
            if best:
                config_df = df[df['config_id'] == best['config_id']]

                if 'episode' in config_df.columns and 'reward' in config_df.columns:
                    # Smooth rewards
                    window = 10
                    smoothed = config_df['reward'].rolling(window=window, min_periods=1).mean()

                    ax.plot(config_df['episode'], smoothed,
                           color=colors[algorithm], linewidth=2, label=algorithm.upper())
                    ax.fill_between(config_df['episode'],
                                   config_df['reward'].rolling(window=window).min(),
                                   config_df['reward'].rolling(window=window).max(),
                                   color=colors[algorithm], alpha=0.2)

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title(f'{algorithm.upper()} Learning Curve', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=300, bbox_inches='tight')
    print(f"Saved learning curves to {output_dir}/learning_curves.png")
    plt.close()


def plot_algorithm_comparison(results: Dict[str, pd.DataFrame], output_dir: str):
    """
    Create bar chart comparing algorithms.

    Args:
        results: Dictionary mapping algorithm names to DataFrames
        output_dir: Directory to save plots
    """
    comparison_data = []

    for algorithm, df in results.items():
        if df.empty:
            continue

        best = find_best_config(df)
        if best:
            comparison_data.append({
                'Algorithm': algorithm.upper(),
                'Mean Reward': best['mean_reward'],
                'Std Reward': best['std_reward'],
                'Triage Accuracy': best['triage_accuracy']
            })

    if not comparison_data:
        print("No data to plot comparison")
        return

    comp_df = pd.DataFrame(comparison_data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Algorithm Performance Comparison', fontsize=16, fontweight='bold')

    # Plot 1: Mean Reward
    ax1 = axes[0]
    bars1 = ax1.bar(comp_df['Algorithm'], comp_df['Mean Reward'],
                    yerr=comp_df['Std Reward'],
                    capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax1.set_ylabel('Mean Reward', fontsize=12)
    ax1.set_title('Mean Episode Reward', fontsize=14)
    ax1.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10)

    # Plot 2: Triage Accuracy
    ax2 = axes[1]
    bars2 = ax2.bar(comp_df['Algorithm'], comp_df['Triage Accuracy'],
                    alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax2.set_ylabel('Triage Accuracy (%)', fontsize=12)
    ax2.set_title('Triage Accuracy', fontsize=14)
    ax2.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'algorithm_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved algorithm comparison to {output_dir}/algorithm_comparison.png")
    plt.close()


def generate_summary_report(results: Dict[str, pd.DataFrame], output_path: str):
    """
    Generate comprehensive summary report.

    Args:
        results: Dictionary mapping algorithm names to DataFrames
        output_path: Path to save summary CSV
    """
    # Create comparison table
    comparison_df = create_comparison_table(results)

    # Save to CSV
    comparison_df.to_csv(output_path, index=False)
    print(f"\nSummary saved to: {output_path}")

    # Print to console
    print("\n" + "="*80)
    print("ALGORITHM COMPARISON SUMMARY")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Aggregate RL training results"
    )

    parser.add_argument(
        '--logs_dir',
        type=str,
        default='logs',
        help='Directory containing algorithm logs'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='evaluation',
        help='Directory to save output files'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)

    print("Loading results...")

    # Load results for each algorithm
    algorithms = ['dqn', 'ppo', 'a2c', 'reinforce']
    results = {}

    for algorithm in algorithms:
        print(f"  Loading {algorithm.upper()}...")
        df = load_csv_results(algorithm, args.logs_dir)
        if not df.empty:
            results[algorithm] = df
            print(f"    Found {len(df)} records")
        else:
            print(f"    No results found")

    if not results:
        print("Error: No results found in logs directory")
        return

    # Generate summary report
    summary_path = os.path.join(args.output_dir, 'results_summary.csv')
    generate_summary_report(results, summary_path)

    # Generate plots
    plots_dir = os.path.join(args.output_dir, 'plots')
    print("\nGenerating plots...")

    plot_learning_curves(results, plots_dir)
    plot_algorithm_comparison(results, plots_dir)

    print("\nAggregation complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
