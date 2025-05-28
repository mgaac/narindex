#!/usr/bin/env python3
"""
Message Passing Graph Neural Networks (MPGNN) Experiment Runner

Usage:
    python experiment.py --total-steps 2000 --aggregation sum
    python experiment.py --total-steps 1000 --eval-interval 25 --mp-layers 3
"""

import argparse
import os
import sys
from pathlib import Path

# Add utils directory to path for data_loading import
sys.path.append('../../utils/datasets/CORA/scripts')
from data_loading import load_data

from model import MPNN, AggregationFunction


def print_experiment_header(**config):
    """Print MPGNN experiment header with configuration."""
    print(f"\n{'='*70}")
    print(f"Message Passing Graph Neural Networks (MPGNN) Experiment")
    print(f"{'='*70}")
    for key, value in config.items():
        print(f"{key}: {value}")
    print(f"{'='*70}")


def run_experiment(args):
    """Run MPGNN training experiment."""
    
    # Map aggregation string to enum
    agg_map = {
        'sum': AggregationFunction.SUM,
        'avg': AggregationFunction.AVG,
        'max': AggregationFunction.MAX,
        'min': AggregationFunction.MIN
    }
    aggregation_enum = agg_map[args.aggregation]
    
    # Configuration
    model_config = {
        'num_nodes': 2708,
        'embedding_dim': 1433,
        'dim_proj': args.hidden_dim,
        'dropout_prob': args.dropout,
        'skip_connections': args.skip_connections,
        'aggregation_fn': aggregation_enum,
        'num_mp_layers': args.mp_layers,
        'num_out_layers': 1,
        'num_classes': 7
    }
    
    training_config = {
        'total_steps': args.total_steps,
        'learning_rate': args.learning_rate,
        'eval_interval': args.eval_interval,
        'patience': args.patience,
        'aggregation': args.aggregation,
        'mp_layers': args.mp_layers
    }
    
    experiment_config = {
        'total_steps': args.total_steps,
        'learning_rate': args.learning_rate,
        'eval_interval': args.eval_interval,
        'dataset': 'CORA (Citation Network)',
        'architecture': f"{args.mp_layers} MP layers, {args.hidden_dim}d hidden",
        'aggregation': f"{args.aggregation.upper()} aggregation function",
        'regularization': f"dropout={args.dropout}, skip_conn={args.skip_connections}"
    }
    
    print_experiment_header(**experiment_config)
    
    # Import and call training function
    from train import train_mpgnn
    
    # Count parameters
    import mlx.core as mx
    
    temp_model = MPNN(**model_config)
    mx.eval(temp_model.parameters())
    
    def count_parameters(params):
        total = 0
        if isinstance(params, dict):
            for v in params.values():
                if hasattr(v, 'size'):
                    total += v.size
                elif isinstance(v, dict):
                    total += count_parameters(v)
        return total
    
    param_count = count_parameters(temp_model.parameters())
    print(f"Model initialized with {param_count:,} parameters")
    
    # Display dataset statistics and train
    data_path = "../../utils/datasets/CORA/data"
    results = train_mpgnn(model_config, training_config, data_path)
    
    # Print dataset statistics
    stats = results['dataset_stats']
    print(f"\nDataset Statistics:")
    print(f"  Nodes: {stats['nodes']:,}")
    print(f"  Features per node: {stats['features']:,}")
    print(f"  Training nodes: {stats['train_nodes']:,}")
    print(f"  Validation nodes: {stats['val_nodes']:,}")
    print(f"  Test nodes: {stats['test_nodes']:,}")
    
    # Print final results
    print(f"\n{'='*70}")
    print(f"MPGNN Training Completed!")
    print(f"{'='*70}")
    print(f"Final Results:")
    print(f"  Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"  Test Loss: {results['test_loss']:.4f}")
    print(f"  Best Validation Loss: {results['best_val_loss']:.4f} (step {results['best_step']})")
    print(f"  Total Steps: {results['total_steps']}")
    print(f"  Evaluation Interval: {results['eval_interval']}")
    print(f"  Aggregation Function: {results['aggregation_function'].upper()}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Message Passing Graph Neural Networks (MPGNN) Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Different aggregation functions
  python experiment.py --aggregation sum --total-steps 2000
  python experiment.py --aggregation avg --total-steps 2000
  python experiment.py --aggregation max --total-steps 2000
  
  # Architecture exploration
  python experiment.py --mp-layers 3 --hidden-dim 16
  
  # Training configuration
  python experiment.py --total-steps 1000 --eval-interval 25 --patience 5
  
  # Regularization study
  python experiment.py --dropout 0.3 --no-skip-connections
  
MPGNN Architecture:
  - Message passing framework for node representation learning
  - Multiple aggregation functions (SUM, AVG, MAX, MIN)
  - Skip connections for better gradient flow
  - Configurable number of message passing layers
  - Step-based training with validation-based early stopping
        """
    )
    
    # Training parameters
    parser.add_argument('--total-steps', type=int, default=2000,
                        help='Total number of training steps (default: 2000)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience in evaluation intervals (default: 10)')
    parser.add_argument('--eval-interval', type=int, default=50,
                        help='Number of steps between evaluations (default: 50)')
    
    # MPGNN-specific architecture
    parser.add_argument('--aggregation', choices=['sum', 'avg', 'max', 'min'], default='max',
                        help='Aggregation function for message passing (default: max)')
    parser.add_argument('--mp-layers', type=int, default=1,
                        help='Number of message passing layers (default: 1)')
    parser.add_argument('--hidden-dim', type=int, default=8,
                        help='Hidden dimension (default: 8)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability (default: 0.5)')
    parser.add_argument('--skip-connections', action='store_true', default=True,
                        help='Use skip connections (default: True)')
    parser.add_argument('--no-skip-connections', action='store_false', dest='skip_connections',
                        help='Disable skip connections')
    
    args = parser.parse_args()
    
    try:
        results = run_experiment(args)
        return 0
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 