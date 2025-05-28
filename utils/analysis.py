"""
Analysis utilities for Neural Algorithmic Reasoning implementations.
Provides functions for computing training statistics, visualizations, and model analysis.
"""

import numpy as np
import pandas as pd
import json
from mlx.utils import tree_flatten
from typing import Dict, Any, List, Tuple


def compute_activation_stats(activations: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute activation statistics for model analysis.
    
    Args:
        activations: Dictionary of activation tensors
        
    Returns:
        Dictionary of computed statistics
    """
    stats = {}
    for act_key, act_tensor in activations.items():
        # Convert MLX array to NumPy array
        act_np = np.array(act_tensor)
        hist, _ = np.histogram(act_np, bins=10)
        stats[f"{act_key}_histogram"] = hist.tolist()
        stats[f"{act_key}_std"] = float(np.std(act_np))
        stats[f"{act_key}_norm"] = float(np.linalg.norm(act_np))
    return stats


def compute_gradient_stats(grads: Any) -> Dict[str, Any]:
    """
    Compute gradient statistics using tree_flatten.
    
    Args:
        grads: Gradient tree structure
        
    Returns:
        Dictionary of gradient statistics
    """
    stats = {}
    flat_grads = tree_flatten(grads)
    for key, grad in flat_grads:
        grad_np = np.array(grad)
        stats[f"{key}_grad_norm"] = float(np.linalg.norm(grad_np))
        stats[f"{key}_grad_variance"] = float(np.var(grad_np))
        hist, _ = np.histogram(grad_np, bins=10)
        stats[f"{key}_grad_histogram"] = hist.tolist()
    return stats


def compute_param_stats(params: Any) -> Dict[str, Any]:
    """
    Compute parameter statistics using tree_flatten.
    
    Args:
        params: Parameter tree structure
        
    Returns:
        Dictionary of parameter statistics
    """
    stats = {}
    flat_params = tree_flatten(params)
    for key, param in flat_params:
        param_np = np.array(param)
        stats[f"{key}_weight_norm"] = float(np.linalg.norm(param_np))
        hist, _ = np.histogram(param_np, bins=10)
        stats[f"{key}_weight_histogram"] = hist.tolist()
    return stats


def compute_update_ratios(prev_params: Any, current_params: Any) -> Dict[str, float]:
    """
    Compute weight update ratios by comparing parameters.
    
    Args:
        prev_params: Previous parameter values
        current_params: Current parameter values
        
    Returns:
        Dictionary of update ratios
    """
    update_ratios = {}
    flat_prev = tree_flatten(prev_params)
    flat_curr = tree_flatten(current_params)
    
    for (key_prev, prev_val), (key_curr, curr_val) in zip(flat_prev, flat_curr):
        prev_np = np.array(prev_val)
        curr_np = np.array(curr_val)
        update = np.linalg.norm(curr_np - prev_np)
        param_norm = np.linalg.norm(prev_np) + 1e-8
        update_ratios[f"{key_curr}_update_ratio"] = float(update / param_norm)
    
    return update_ratios


def get_table_scalar_metric(csv_file: str, column_name: str, metric_name: str) -> pd.DataFrame:
    """
    Extract scalar metric from CSV log as DataFrame.
    
    Args:
        csv_file: Path to CSV log file
        column_name: Column to extract data from
        metric_name: Name for the metric row
        
    Returns:
        DataFrame with metric data
    """
    df = pd.read_csv(csv_file)
    df['Iteration'] = df['Iteration'].astype(int)
    data = {}
    for _, row in df.iterrows():
        iteration = row['Iteration']
        data[iteration] = row[column_name]
    
    df_table = pd.DataFrame([data], index=[metric_name])
    df_table = df_table.reindex(sorted(df_table.columns), axis=1)
    return df_table


def get_table_metric(csv_file: str, column_name: str, metric_suffix: str) -> pd.DataFrame:
    """
    Extract layered metrics from CSV log.
    
    Args:
        csv_file: Path to CSV log file
        column_name: Column containing JSON data
        metric_suffix: Suffix to filter metrics by
        
    Returns:
        DataFrame with rows=layers, columns=iterations
    """
    df = pd.read_csv(csv_file)
    df['Iteration'] = df['Iteration'].astype(int)
    table = {}

    for _, row in df.iterrows():
        iteration = row["Iteration"]
        try:
            data = json.loads(row[column_name])
        except (TypeError, json.JSONDecodeError):
            continue
        
        for key, value in data.items():
            if key.endswith(metric_suffix):
                param_name = key[:-len(metric_suffix)-1]
                if param_name not in table:
                    table[param_name] = {}
                table[param_name][iteration] = value

    df_table = pd.DataFrame(table).T
    df_table = df_table.reindex(sorted(df_table.columns), axis=1)
    return df_table


def print_table_with_title(df_table: pd.DataFrame, title: str, max_cols: int = 10):
    """
    Print DataFrame with title and column truncation if needed.
    
    Args:
        df_table: DataFrame to print
        title: Title for the table
        max_cols: Maximum columns to show before truncating
    """
    print("\n" + title)
    print("=" * len(title))
    
    if df_table.empty:
        print("No data available.\n")
        return

    num_cols = len(df_table.columns)
    if num_cols > max_cols:
        half = max_cols // 2
        first_cols = list(df_table.columns[:half])
        last_cols = list(df_table.columns[-half:])
        df_trunc = pd.concat([df_table[first_cols], df_table[last_cols]], axis=1)
        df_trunc.insert(half, '...', ['...'] * len(df_trunc))
        print(df_trunc.to_string())
    else:
        print(df_table.to_string())
    print("\n")


class TrainingLogger:
    """
    Standardized training logger for consistent metrics tracking.
    """
    
    def __init__(self, log_file: str = "training_log.csv"):
        """Initialize training logger."""
        self.log_file = log_file
        self.metrics_history = []
        
    def log_step(self, step: int, metrics: Dict[str, Any]):
        """Log metrics for a training step."""
        log_entry = {"step": step, **metrics}
        self.metrics_history.append(log_entry)
        
    def save_logs(self):
        """Save accumulated logs to CSV file."""
        if self.metrics_history:
            df = pd.DataFrame(self.metrics_history)
            df.to_csv(self.log_file, index=False)
            print(f"Training logs saved to {self.log_file}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of training metrics."""
        if not self.metrics_history:
            return {}
            
        df = pd.DataFrame(self.metrics_history)
        summary = {}
        
        for col in df.columns:
            if col != 'step' and df[col].dtype in ['float64', 'int64']:
                summary[f"{col}_final"] = df[col].iloc[-1]
                summary[f"{col}_mean"] = df[col].mean()
                summary[f"{col}_std"] = df[col].std()
                
        return summary


# Convenience functions for specific metrics
def table_weight_norm(csv_file: str) -> pd.DataFrame:
    """Extract weight norms from training log."""
    return get_table_metric(csv_file, "Parameter Stats", "weight_norm")


def table_loss(csv_file: str) -> pd.DataFrame:
    """Extract training loss from log."""
    return get_table_scalar_metric(csv_file, "Training Loss", "loss")


def table_accuracy(csv_file: str) -> pd.DataFrame:
    """Extract accuracy scores from log."""
    return get_table_scalar_metric(csv_file, "Accuracy Score", "accuracy")


def generate_training_report(csv_file: str) -> None:
    """Generate comprehensive training analysis report."""
    print("Neural Algorithmic Reasoning - Training Analysis Report")
    print("=" * 55)
    
    # Loss analysis
    loss_table = table_loss(csv_file)
    print_table_with_title(loss_table, "Training Loss Over Time")
    
    # Accuracy analysis
    try:
        acc_table = table_accuracy(csv_file)
        print_table_with_title(acc_table, "Accuracy Over Time")
    except:
        print("No accuracy data available")
    
    # Weight analysis
    try:
        weight_table = table_weight_norm(csv_file)
        print_table_with_title(weight_table, "Weight Norms Over Time")
    except:
        print("No weight norm data available")
    
    # Gradient analysis
    try:
        grad_table = get_table_metric(csv_file, "Gradient Stats", "grad_norm")
        print_table_with_title(grad_table, "Gradient Norms Over Time")
    except:
        print("No gradient data available")