"""
Graph Attention Network (GAT) training script.
Uses standardized utilities for consistent training patterns.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from model import GAT
from utils.common import standard_arg_parser, cross_entropy_loss_fn, accuracy_fn, print_model_summary, setup_reproducibility
from utils.datasets.CORA.cora import load_cora_data, get_cora_config


def create_gat_arg_parser():
    """Create argument parser with GAT-specific options."""
    parser = standard_arg_parser("Train a Graph Attention Network on the CORA dataset")
    
    # Override default data path
    parser.set_defaults(data="../../utils/datasets/CORA/data")
    
    # GAT-specific arguments
    parser.add_argument(
        "--dim-proj", type=int, default=8,
        help="Projection dimension per attention head"
    )
    parser.add_argument(
        "--num-att-heads", type=int, default=8,
        help="Number of attention heads"
    )
    parser.add_argument(
        "--num-layers", type=int, default=1,
        help="Number of GAT layers"
    )
    parser.add_argument(
        "--skip-connections", action="store_true", default=True,
        help="Use skip connections"
    )
    parser.add_argument(
        "--num-out-layers", type=int, default=1,
        help="Number of output layers"
    )
    
    return parser


def train_gat(model_config, training_config, data_path):
    """
    Train GAT model - function for experiment scripts.
    
    Args:
        model_config: Dictionary of model configuration parameters
        training_config: Dictionary of training configuration parameters  
        data_path: Path to dataset
        
    Returns:
        Dictionary of training results
    """
    # Setup reproducibility
    setup_reproducibility()
    
    # Load CORA dataset
    print("Loading CORA dataset...")
    node_embeddings, connection_matrix, labels, train_mask, test_mask = load_cora_data(data_path)
    
    # Initialize model and optimizer
    print("\nInitializing model...")
    model = GAT(**model_config)
    mx.eval(model.parameters())
    optimizer = optim.Adam(learning_rate=training_config['learning_rate'])
    
    # Print model summary
    print_model_summary(model, "GAT")
    
    # Prepare data
    data = (node_embeddings, connection_matrix)
    train_data = (data, labels, train_mask)
    test_data = (data, labels, test_mask)
    
    # Create training loop
    from utils.common import TrainingLoop
    trainer = TrainingLoop(model, optimizer, cross_entropy_loss_fn, accuracy_fn)
    
    # Training
    print(f"\nStarting training for {training_config['total_steps']} steps...")
    results = trainer.train(train_data, test_data, training_config['total_steps'], training_config['eval_interval'])
    
    # Prepare return results
    dataset_stats = {
        'nodes': node_embeddings.shape[0],
        'features': node_embeddings.shape[1],
        'train_nodes': int(train_mask.sum()),
        'val_nodes': 0,  # No separate validation set in current implementation
        'test_nodes': int(test_mask.sum())
    }
    
    return {
        'test_accuracy': results['final_accuracy'],
        'test_loss': results['final_loss'],
        'best_val_loss': results['final_loss'],  # Using final loss as best
        'best_step': training_config['total_steps'],
        'total_steps': training_config['total_steps'],
        'eval_interval': training_config['eval_interval'],
        'dataset_stats': dataset_stats
    }


def main():
    """Main training function."""
    args = create_gat_arg_parser().parse_args()
    
    # Setup reproducibility
    setup_reproducibility()
    
    # Load CORA dataset
    print("Loading CORA dataset...")
    node_embeddings, connection_matrix, labels, train_mask, test_mask = load_cora_data(args.data)
    cora_config = get_cora_config()
    
    # Model configuration
    model_config = {
        'num_nodes': cora_config['num_nodes'],
        'dim_embed': cora_config['num_features'],
        'dim_proj': args.dim_proj,
        'num_att_heads': args.num_att_heads,
        'num_layers': args.num_layers,
        'skip_connections': args.skip_connections,
        'dropout_prob': args.dropout,
        'num_out_layers': args.num_out_layers,
        'num_out_classes': cora_config['num_classes']
    }
    
    # Initialize model and optimizer
    print("\nInitializing model...")
    model = GAT(**model_config)
    mx.eval(model.parameters())
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    
    # Print model summary
    print_model_summary(model, "GAT")
    
    # Prepare data
    data = (node_embeddings, connection_matrix)
    train_data = (data, labels, train_mask)
    test_data = (data, labels, test_mask)
    
    # Create training loop
    from utils.common import TrainingLoop
    trainer = TrainingLoop(model, optimizer, cross_entropy_loss_fn, accuracy_fn)
    
    # Training
    print(f"\nStarting training for {args.num_steps} steps...")
    results = trainer.train(train_data, test_data, args.num_steps, args.eval_interval)
    
    # Final evaluation
    print(f"\nTraining completed!")
    print(f"Final training loss: {results['final_loss']:.4f}")
    print(f"Final test accuracy: {results['final_accuracy']:.4f}")
    
    # Save model summary
    print(f"\nModel configuration:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
