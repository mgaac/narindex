#!/usr/bin/env python3
"""Neural Execution of Graph Algorithms (NGE) Experiment Runner

Paper: Neural Execution of Graph Algorithms

Learns to execute Bellman-Ford (shortest paths) and BFS (reachability)
algorithms using a shared encoder-processor-decoder architecture.

Usage:
    python experiment.py --num-epochs 100
    python experiment.py --num-epochs 500 --learning-rate 1e-5 --batch-size 10
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import mlx.core as mx
import mlx.optimizers as optim
from tqdm import tqdm

from model import NGE, AggregationFunction
from train import train_epoch, evaluate_model
from data_loading import load_dataset
from nge_utils import count_parameters
from utils.common import setup_reproducibility


def print_experiment_header(config: dict):
    """Print standardized experiment header."""
    print(f"\n{'='*70}")
    print("Neural Execution of Graph Algorithms (NGE)")
    print("Learning: Bellman-Ford (Shortest Paths) & BFS (Reachability)")
    print(f"{'='*70}")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"{'='*70}\n")


def run_experiment(args):
    """Run NGE training experiment."""
    setup_reproducibility()
    
    # Map aggregation function
    agg_map = {
        'sum': AggregationFunction.SUM,
        'avg': AggregationFunction.AVG,
        'max': AggregationFunction.MAX,
        'min': AggregationFunction.MIN
    }
    aggregation_fn = agg_map[args.aggregation]
    
    # Model configuration
    model_config = {
        'embedding_dim': args.hidden_dim,
        'residual_connections': args.residual_connections,
        'aggregation_fn': aggregation_fn,
        'num_mp_layers': args.num_layers,
        'dropout': args.dropout,
        'num_predecessor_layers': args.num_predecessor_layers,
        'num_update_layers': args.num_update_layers
    }
    
    # Experiment configuration for display
    experiment_config = {
        'Dataset': 'Generated Graphs (BF + BFS execution traces)',
        'Algorithms': 'Bellman-Ford, BFS',
        'Architecture': f"{args.num_layers} MP layers, {args.hidden_dim}d embedding",
        'Aggregation': f"{args.aggregation.upper()} function",
        'Training Epochs': args.num_epochs,
        'Learning Rate': f"{args.start_lr} -> {args.end_lr}",
        'Batch Size': args.batch_size,
        'Max Grad Norm': args.max_grad_norm,
        'BF Predecessor Weight': args.bf_pred_alpha,
        'Regularization': f"dropout={args.dropout}, residual={args.residual_connections}"
    }
    
    print_experiment_header(experiment_config)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = load_dataset(os.path.join(args.data, 'train_dataset.npz'))
    val_dataset = load_dataset(os.path.join(args.data, 'val_dataset.npz'))
    test_dataset = load_dataset(os.path.join(args.data, 'test_dataset.npz'))
    
    print(f"  Training graphs: {len(train_dataset):,}")
    print(f"  Validation graphs: {len(val_dataset):,}")
    print(f"  Test graphs: {len(test_dataset):,}")
    
    # Initialize model
    print("\nInitializing model...")
    model = NGE(**model_config)
    model.train()
    
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count:,}")
    
    # Set up learning rate schedule
    total_steps = args.num_epochs * (len(train_dataset) / args.batch_size)
    decay_steps = int(total_steps * args.decay_ratio)
    
    lr_schedule = optim.cosine_decay(
        init=args.start_lr,
        decay_steps=decay_steps,
        end=args.end_lr
    )
    
    optimizer = optim.Adam(learning_rate=lr_schedule)
    
    # Training tracking
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    train_history = []
    val_history = []
    
    print(f"\nStarting training for {args.num_epochs} epochs...")
    
    # Training loop
    pbar = tqdm(range(args.num_epochs), desc="Training NGE", unit="epoch")
    
    for epoch in pbar:
        # Training phase
        avg_train_loss, avg_aux_losses, avg_per_head_grads = train_epoch(
            model=model,
            optimizer=optimizer,
            dataset=train_dataset,
            embedding_dim=args.hidden_dim,
            batch_size=args.batch_size,
            max_grad_norm=args.max_grad_norm,
            bf_pred_alpha=args.bf_pred_alpha,
            label_smoothing=args.label_smoothing
        )
        
        train_history.append(float(avg_train_loss))
        
        # Validation phase (every 10 epochs)
        if (epoch + 1) % 10 == 0 or epoch == args.num_epochs - 1:
            val_aux_losses, val_loss, val_accuracies = evaluate_model(
                model, val_dataset, args.hidden_dim
            )
            
            val_history.append(float(val_loss))
            
            # Early stopping check
            if float(val_loss) < best_val_loss:
                best_val_loss = float(val_loss)
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            pbar.set_postfix({
                'train': f"{float(avg_train_loss):.4f}",
                'val': f"{float(val_loss):.4f}",
                'best': f"{best_val_loss:.4f}",
                'bf_dist_acc': f"{float(val_accuracies[0]):.3f}",
                'bfs_acc': f"{float(val_accuracies[2]):.3f}"
            })
            
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        else:
            pbar.set_postfix({
                'train': f"{float(avg_train_loss):.4f}",
                'lr': f"{float(optimizer.learning_rate):.2e}"
            })
    
    # Final evaluation
    print(f"\n{'='*70}")
    print("Training Complete")
    print(f"{'='*70}")
    print(f"  Best Validation Loss: {best_val_loss:.4f} (epoch {best_epoch + 1})")
    print(f"  Final Training Loss: {train_history[-1]:.4f}")
    
    # Test set evaluation
    print("\nTest Set Evaluation:")
    test_aux_losses, test_loss, test_accuracies = evaluate_model(
        model, test_dataset, args.hidden_dim
    )
    
    print(f"  Test Loss: {float(test_loss):.4f}")
    print(f"  Accuracies:")
    print(f"    BF Distance: {float(test_accuracies[0]):.3f}")
    print(f"    BF Predecessor: {float(test_accuracies[1]):.3f}")
    print(f"    BFS State: {float(test_accuracies[2]):.3f}")
    print(f"    BF Termination: {float(test_accuracies[3]):.3f}")
    print(f"    BFS Termination: {float(test_accuracies[4]):.3f}")
    
    print(f"{'='*70}\n")
    
    return {
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'final_train_loss': train_history[-1],
        'test_loss': float(test_loss),
        'test_accuracies': {
            'bf_distance': float(test_accuracies[0]),
            'bf_predecessor': float(test_accuracies[1]),
            'bfs_state': float(test_accuracies[2]),
            'bf_termination': float(test_accuracies[3]),
            'bfs_termination': float(test_accuracies[4])
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Neural Execution of Graph Algorithms (NGE) Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiment.py --num-epochs 100
  python experiment.py --num-epochs 500 --batch-size 10 --start-lr 1e-5
  python experiment.py --num-layers 3 --hidden-dim 64
        """
    )
    
    # Training parameters
    parser.add_argument('--num-epochs', type=int, default=500,
                        help='Number of training epochs (default: 500)')
    parser.add_argument('--start-lr', type=float, default=1e-5,
                        help='Starting learning rate (default: 1e-5)')
    parser.add_argument('--end-lr', type=float, default=1e-5,
                        help='Ending learning rate (default: 1e-5)')
    parser.add_argument('--decay-ratio', type=float, default=0.005,
                        help='LR decay ratio (default: 0.005)')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Gradient accumulation batch size (default: 10)')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                        help='Maximum gradient norm (default: 1.0)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience in validation cycles (default: 50)')
    
    # Loss parameters
    parser.add_argument('--bf-pred-alpha', type=float, default=1.0,
                        help='Weight for BF predecessor loss (default: 1.0)')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='Label smoothing for cross-entropy (default: 0.0)')
    
    # Architecture parameters
    parser.add_argument('--aggregation', choices=['sum', 'avg', 'max', 'min'], default='max',
                        help='Aggregation function (default: max)')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of message passing layers (default: 2)')
    parser.add_argument('--hidden-dim', type=int, default=32,
                        help='Node embedding dimension (default: 32)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability (default: 0.1)')
    parser.add_argument('--residual-connections', action='store_true', default=True,
                        help='Use residual connections (default: True)')
    parser.add_argument('--no-residual-connections', action='store_false', dest='residual_connections',
                        help='Disable residual connections')
    parser.add_argument('--num-predecessor-layers', type=int, default=5,
                        help='Number of predecessor prediction layers (default: 5)')
    parser.add_argument('--num-update-layers', type=int, default=1,
                        help='Number of update layers per MP layer (default: 1)')
    
    # Data parameters
    parser.add_argument('--data', type=str, default='data',
                        help='Path to dataset directory')
    
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
