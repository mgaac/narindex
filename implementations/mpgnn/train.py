"""
Message Passing Neural Network (MPNN) Training Module

Provides training utilities for the MPNN model on node classification tasks.
For full experiments, use experiment.py instead.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import mlx.core as mx
import mlx.optimizers as optim

from model import MPNN, AggregationFunction
from utils.common import (
    cross_entropy_loss_fn, accuracy_fn, print_model_summary, 
    setup_reproducibility, TrainingLoop
)
from utils.datasets.CORA.cora import load_cora_data, get_cora_config


def train_mpnn(model_config: dict, training_config: dict, data_path: str) -> dict:
    """
    Train MPNN model.
    
    Args:
        model_config: Model configuration parameters
        training_config: Training configuration parameters  
        data_path: Path to dataset
        
    Returns:
        Dictionary of training results
    """
    setup_reproducibility()
    
    # Load CORA dataset
    print("Loading CORA dataset...")
    node_embeddings, connection_matrix, labels, train_mask, test_mask = load_cora_data(data_path)
    
    # Initialize model
    print("Initializing model...")
    model = MPNN(**model_config)
    mx.eval(model.parameters())
    optimizer = optim.Adam(learning_rate=training_config['learning_rate'])
    
    print_model_summary(model, "MPNN")
    
    # Prepare data
    data = (node_embeddings, connection_matrix)
    train_data = (data, labels, train_mask)
    test_data = (data, labels, test_mask)
    
    # Training
    trainer = TrainingLoop(model, optimizer, cross_entropy_loss_fn, accuracy_fn)
    
    print(f"\nStarting training for {training_config['num_steps']} steps...")
    results = trainer.train(
        train_data, test_data, 
        training_config['num_steps'], 
        training_config['eval_interval']
    )
    
    # Prepare results
    dataset_stats = {
        'nodes': node_embeddings.shape[0],
        'features': node_embeddings.shape[1],
        'train_nodes': int(train_mask.sum()),
        'val_nodes': 0,
        'test_nodes': int(test_mask.sum())
    }
    
    return {
        'test_accuracy': results['final_accuracy'],
        'test_loss': results['final_loss'],
        'aggregation_function': training_config.get('aggregation', 'unknown'),
        'dataset_stats': dataset_stats
    }


if __name__ == "__main__":
    print("Use experiment.py for full experiment runs.")
    print("This module provides train_mpnn() for programmatic use.")
