"""
Evaluation script: Fetch metrics from WandB and generate comparison figures.
Independent script, not called from main.py.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def fetch_wandb_run_data(entity: str, project: str, run_id: str) -> Dict[str, Any]:
    """
    Fetch run history, summary, and config from WandB API.
    
    Args:
        entity: WandB entity
        project: WandB project
        run_id: Run ID
        
    Returns:
        Dictionary with history, summary, and config
    """
    api = wandb.Api()
    
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        
        # Get history (time series metrics)
        history = run.history()
        
        # Get summary (final metrics)
        summary = dict(run.summary)
        
        # Get config
        config = dict(run.config)
        
        return {
            'run_id': run_id,
            'history': history,
            'summary': summary,
            'config': config,
        }
    except Exception as e:
        print(f"Warning: Could not fetch WandB data for {run_id}: {e}")
        return None


def load_local_results(results_dir: Path, run_id: str) -> Dict[str, Any]:
    """
    Load results from local JSON file as fallback.
    
    Args:
        results_dir: Results directory
        run_id: Run ID
        
    Returns:
        Results dictionary
    """
    results_path = results_dir / run_id / "results.json"
    if results_path.exists():
        with open(results_path, 'r') as f:
            return json.load(f)
    return None


def export_per_run_metrics(results_dir: Path, run_id: str, data: Dict[str, Any]):
    """
    Export per-run metrics to JSON.
    
    Args:
        results_dir: Results directory
        run_id: Run ID
        data: Run data from WandB or local
    """
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_path = run_dir / "metrics.json"
    
    # Extract key metrics
    if 'summary' in data:
        metrics = data['summary']
    else:
        metrics = data
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Exported metrics: {metrics_path}")


def create_per_run_figures(results_dir: Path, run_id: str, data: Dict[str, Any]):
    """
    Create per-run figures (if applicable for this experiment).
    For inference-only tasks, there may not be training curves.
    
    Args:
        results_dir: Results directory
        run_id: Run ID
        data: Run data
    """
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Since this is inference-only, we mainly have summary metrics
    # Create a simple bar chart of key metrics
    
    if 'summary' in data:
        metrics = data['summary']
    else:
        metrics = data
    
    # Extract numeric metrics
    metric_names = []
    metric_values = []
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not key.startswith('_'):
            metric_names.append(key)
            metric_values.append(value)
    
    if len(metric_names) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(metric_names, metric_values)
        ax.set_xlabel('Value')
        ax.set_title(f'Metrics for {run_id}')
        plt.tight_layout()
        
        figure_path = run_dir / f"metrics_summary_{run_id}.pdf"
        plt.savefig(figure_path, format='pdf', bbox_inches='tight')
        plt.close()
        
        print(f"Created figure: {figure_path}")


def create_comparison_figures(results_dir: Path, all_run_data: Dict[str, Dict]):
    """
    Create comparison figures across all runs.
    
    Args:
        results_dir: Results directory
        all_run_data: Dictionary mapping run_id to run data
    """
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract common metrics across runs
    common_metrics = set()
    for run_id, data in all_run_data.items():
        if 'summary' in data:
            metrics = data['summary']
        else:
            metrics = data
        
        for key in metrics.keys():
            if isinstance(metrics[key], (int, float)) and not key.startswith('_'):
                common_metrics.add(key)
    
    # Create comparison bar charts for each metric
    for metric_name in common_metrics:
        run_ids = []
        values = []
        
        for run_id, data in all_run_data.items():
            if 'summary' in data:
                metrics = data['summary']
            else:
                metrics = data
            
            if metric_name in metrics:
                run_ids.append(run_id)
                values.append(metrics[metric_name])
        
        if len(run_ids) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Use different colors for proposed vs comparative
            colors = []
            for rid in run_ids:
                if 'proposed' in rid:
                    colors.append('tab:blue')
                else:
                    colors.append('tab:orange')
            
            ax.bar(range(len(run_ids)), values, color=colors)
            ax.set_xticks(range(len(run_ids)))
            ax.set_xticklabels(run_ids, rotation=45, ha='right')
            ax.set_ylabel(metric_name)
            ax.set_title(f'Comparison: {metric_name}')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='tab:blue', label='Proposed'),
                Patch(facecolor='tab:orange', label='Baseline')
            ]
            ax.legend(handles=legend_elements)
            
            plt.tight_layout()
            
            # Safe filename
            safe_metric_name = metric_name.replace('/', '_').replace(' ', '_')
            figure_path = comparison_dir / f"comparison_{safe_metric_name}.pdf"
            plt.savefig(figure_path, format='pdf', bbox_inches='tight')
            plt.close()
            
            print(f"Created comparison figure: {figure_path}")


def export_aggregated_metrics(results_dir: Path, all_run_data: Dict[str, Dict], primary_metric: str = 'accuracy'):
    """
    Export aggregated metrics comparing all runs.
    
    Args:
        results_dir: Results directory
        all_run_data: Dictionary mapping run_id to run data
        primary_metric: Primary metric for comparison
    """
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect metrics by run
    metrics_by_run = {}
    for run_id, data in all_run_data.items():
        if 'summary' in data:
            metrics = data['summary']
        else:
            metrics = data
        
        metrics_by_run[run_id] = metrics
    
    # Find best proposed and best baseline
    proposed_runs = {rid: metrics for rid, metrics in metrics_by_run.items() if 'proposed' in rid}
    baseline_runs = {rid: metrics for rid, metrics in metrics_by_run.items() if 'comparative' in rid}
    
    best_proposed = None
    best_proposed_value = -float('inf')
    for run_id, metrics in proposed_runs.items():
        if primary_metric in metrics:
            value = metrics[primary_metric]
            if value > best_proposed_value:
                best_proposed_value = value
                best_proposed = run_id
    
    best_baseline = None
    best_baseline_value = -float('inf')
    for run_id, metrics in baseline_runs.items():
        if primary_metric in metrics:
            value = metrics[primary_metric]
            if value > best_baseline_value:
                best_baseline_value = value
                best_baseline = run_id
    
    # Compute gap
    gap = best_proposed_value - best_baseline_value if (best_proposed and best_baseline) else None
    
    # Create aggregated metrics
    aggregated = {
        'primary_metric': primary_metric,
        'metrics': metrics_by_run,
        'best_proposed': best_proposed,
        'best_proposed_value': best_proposed_value if best_proposed else None,
        'best_baseline': best_baseline,
        'best_baseline_value': best_baseline_value if best_baseline else None,
        'gap': gap,
    }
    
    # Export
    aggregated_path = comparison_dir / "aggregated_metrics.json"
    with open(aggregated_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"\nAggregated metrics:")
    print(f"  Primary metric: {primary_metric}")
    if best_proposed:
        print(f"  Best proposed: {best_proposed} ({best_proposed_value:.4f})")
    if best_baseline:
        print(f"  Best baseline: {best_baseline} ({best_baseline_value:.4f})")
    if gap is not None:
        print(f"  Gap: {gap:.4f}")
    print(f"\nExported: {aggregated_path}")


def main():
    """
    Main evaluation script.
    """
    parser = argparse.ArgumentParser(description="Evaluate and compare experiment runs")
    parser.add_argument('--results_dir', type=str, required=True, help='Results directory')
    parser.add_argument('--run_ids', type=str, required=True, help='JSON list of run IDs')
    parser.add_argument('--wandb_entity', type=str, default='airas', help='WandB entity')
    parser.add_argument('--wandb_project', type=str, default='2026-02-19', help='WandB project')
    parser.add_argument('--primary_metric', type=str, default='accuracy', help='Primary metric for comparison')
    
    args = parser.parse_args()
    
    # Parse run_ids
    run_ids = json.loads(args.run_ids)
    print(f"Evaluating {len(run_ids)} runs: {run_ids}")
    
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch data for each run
    all_run_data = {}
    for run_id in run_ids:
        print(f"\nProcessing {run_id}...")
        
        # Try WandB first
        wandb_data = fetch_wandb_run_data(args.wandb_entity, args.wandb_project, run_id)
        
        if wandb_data:
            all_run_data[run_id] = wandb_data
        else:
            # Fallback to local results
            local_data = load_local_results(results_dir, run_id)
            if local_data:
                all_run_data[run_id] = local_data
            else:
                print(f"Warning: No data found for {run_id}")
                continue
        
        # Export per-run metrics
        export_per_run_metrics(results_dir, run_id, all_run_data[run_id])
        
        # Create per-run figures
        create_per_run_figures(results_dir, run_id, all_run_data[run_id])
    
    # Create comparison figures
    if len(all_run_data) > 1:
        print("\nCreating comparison figures...")
        create_comparison_figures(results_dir, all_run_data)
        
        # Export aggregated metrics
        export_aggregated_metrics(results_dir, all_run_data, args.primary_metric)
    
    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
