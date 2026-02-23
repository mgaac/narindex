#!/usr/bin/env python3
"""
Message Passing Neural Networks (MPNN) Experiment Runner

Paper: "Neural Message Passing for Quantum Chemistry" (Gilmer et al., 2017)
       arXiv:1704.01212v2

Usage:
    python experiment.py --num-steps 2000 --aggregation sum
    python experiment.py --num-steps 1000 --eval-interval 25 --num-layers 3
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import mlx.core as mx
import mlx.optimizers as optim

from model import MPNN, AggregationFunction
from utils.common import (
    cross_entropy_loss_fn, accuracy_fn,
    setup_reproducibility, TrainingLoop
)
from utils.datasets.CORA.cora import load_cora_data, get_cora_config


def print_experiment_header(config: dict):
    """Print standardized experiment header."""
    print(f"\n{'='*70}")
    print("Message Passing Neural Networks (MPNN)")
    print("Paper: arXiv:1704.01212v2")
    print(f"{'='*70}")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"{'='*70}\n")


def count_parameters(model) -> int:
    """Count total model parameters."""
    def _count(params):
        total = 0
        if isinstance(params, dict):
            for v in params.values():
                total += _count(v)
        elif hasattr(params, 'size'):
            total += params.size
        return total
    return _count(model.parameters())


def run_experiment(args):
    """Run MPNN training experiment."""
    setup_reproducibility()
    
    # Map aggregation string to enum
    agg_map = {
        'sum': AggregationFunction.SUM,
        'avg': AggregationFunction.AVG,
        'max': AggregationFunction.MAX,
        'min': AggregationFunction.MIN
    }
    aggregation_fn = agg_map[args.aggregation]
    
    # Load dataset
    print("Loading CORA dataset...")
    cora_config = get_cora_config()
    node_embeddings, connection_matrix, labels, train_mask, test_mask = load_cora_data(args.data)
    
    # Model configuration
    model_config = {
        'num_nodes': cora_config['num_nodes'],
        'embedding_dim': cora_config['num_features'],
        'dim_proj': args.hidden_dim,
        'dropout_prob': args.dropout,
        'skip_connections': args.skip_connections,
        'aggregation_fn': aggregation_fn,
        'num_mp_layers': args.num_layers,
        'num_out_layers': 1,
        'num_classes': cora_config['num_classes']
    }
    
    # Experiment configuration for display
    experiment_config = {
        'Dataset': 'CORA (Citation Network)',
        'Architecture': f"{args.num_layers} MP layers, {args.hidden_dim}d hidden",
        'Aggregation': f"{args.aggregation.upper()} function",
        'Training Steps': args.num_steps,
        'Learning Rate': args.learning_rate,
        'Eval Interval': args.eval_interval,
        'Regularization': f"dropout={args.dropout}, skip_conn={args.skip_connections}"
    }
    
    print_experiment_header(experiment_config)
    
    # Initialize model
    print("Initializing model...")
    model = MPNN(**model_config)
    mx.eval(model.parameters())
    
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count:,}")
    
    # Initialize optimizer
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    
    # Prepare data
    data = (node_embeddings, connection_matrix)
    train_data = (data, labels, train_mask)
    test_data = (data, labels, test_mask)
    
    # Training loop
    trainer = TrainingLoop(model, optimizer, cross_entropy_loss_fn, accuracy_fn)
    
    print(f"\nStarting training for {args.num_steps} steps...")
    results = trainer.train(train_data, test_data, args.num_steps, args.eval_interval)
    
    # Print results
    print(f"\n{'='*70}")
    print("Training Complete")
    print(f"{'='*70}")
    print(f"  Final Loss: {results['final_loss']:.4f}")
    print(f"  Final Accuracy: {results['final_accuracy']:.4f}")
    print(f"{'='*70}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Message Passing Neural Networks (MPNN) Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiment.py --aggregation sum --num-steps 2000
  python experiment.py --aggregation max --num-layers 3 --hidden-dim 16
  python experiment.py --dropout 0.3 --no-skip-connections
        """
    )
    
    # Training parameters
    parser.add_argument('--num-steps', type=int, default=2000,
                        help='Number of training steps (default: 2000)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--eval-interval', type=int, default=50,
                        help='Evaluation interval in steps (default: 50)')
    
    # Architecture parameters
    parser.add_argument('--aggregation', choices=['sum', 'avg', 'max', 'min'], default='max',
                        help='Aggregation function (default: max)')
    parser.add_argument('--num-layers', type=int, default=1,
                        help='Number of message passing layers (default: 1)')
    parser.add_argument('--hidden-dim', type=int, default=8,
                        help='Hidden dimension (default: 8)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability (default: 0.5)')
    parser.add_argument('--skip-connections', action='store_true', default=True,
                        help='Use skip connections (default: True)')
    parser.add_argument('--no-skip-connections', action='store_false', dest='skip_connections',
                        help='Disable skip connections')
    
    # Data parameters
    parser.add_argument('--data', type=str, default='../../utils/datasets/CORA/data',
                        help='Path to CORA dataset')
    
    args = parser.parse_args()
    
    try:
        run_experiment(args)
        return 0
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
