"""
CORA dataset utilities for Neural Algorithmic Reasoning implementations.
Standardized data loading for citation network classification tasks.
"""

import mlx.core as mx
import numpy as np
import pickle
import os
from typing import Tuple


def _pickle_read(path: str):
    """Read pickle file with latin1 encoding for compatibility."""
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def _generate_connection_matrix(graph: dict) -> mx.array:
    """Generate connection matrix from graph adjacency dictionary."""
    edges = [(src, trg) for src, trgs in graph.items() for trg in trgs]
    if not edges:
        raise ValueError("Graph contains no edges")
    
    src_idx, trg_idx = zip(*edges)
    return mx.array([src_idx, trg_idx])


def load_cora_data(data_dir: str = "data/CORA") -> Tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """
    Load CORA citation network dataset.
    
    Args:
        data_dir: Path to CORA dataset directory
        
    Returns:
        Tuple of (node_embeddings, connection_matrix, labels, train_mask, test_mask)
        
    Raises:
        FileNotFoundError: If required data files are missing
        ValueError: If data format is invalid
    """
    required_files = ['allx', 'ally', 'tx', 'ty', 'graph', 'test_index']
    
    # Validate all required files exist
    for filename in required_files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Required CORA file not found: {filepath}")
    
    try:
        # Load pickle files
        allx = _pickle_read(os.path.join(data_dir, 'allx'))
        ally = _pickle_read(os.path.join(data_dir, 'ally'))
        tx = _pickle_read(os.path.join(data_dir, 'tx'))
        ty = _pickle_read(os.path.join(data_dir, 'ty'))
        graph = _pickle_read(os.path.join(data_dir, 'graph'))
        
        # Load test indices
        test_idx = np.loadtxt(os.path.join(data_dir, 'test_index'), dtype=int)
        
        # Convert to MLX arrays
        allx = mx.array(allx.todense())
        ally = mx.array(ally)
        tx = mx.array(tx.todense())
        ty = mx.array(ty)
        test_idx = mx.array(test_idx)
        
        # Combine training and test data
        node_embeddings = mx.concatenate([allx, tx], axis=0)
        labels = mx.concatenate([ally, ty], axis=0)
        connection_matrix = _generate_connection_matrix(graph)
        
        # Create masks using the working approach from data_loading.py
        train_mask = mx.ones([labels.shape[0]]).at[test_idx].add(-1)
        test_mask = mx.zeros([labels.shape[0]]).at[test_idx].add(1)
        
        # Validate data consistency
        num_nodes = node_embeddings.shape[0]
        if labels.shape[0] != num_nodes:
            raise ValueError(f"Node embeddings ({num_nodes}) and labels ({labels.shape[0]}) size mismatch")
        
        if train_mask.shape[0] != num_nodes or test_mask.shape[0] != num_nodes:
            raise ValueError("Mask and node count mismatch")
        
        print(f"CORA dataset loaded: {num_nodes} nodes, {node_embeddings.shape[1]} features, "
              f"{labels.shape[1]} classes")
        print(f"Train nodes: {int(train_mask.sum())}, Test nodes: {int(test_mask.sum())}")
        
        return node_embeddings, connection_matrix, labels, train_mask, test_mask
        
    except Exception as e:
        raise ValueError(f"Error loading CORA data: {str(e)}")


def get_cora_config() -> dict:
    """Get standard CORA dataset configuration parameters."""
    return {
        'num_nodes': 2708,
        'num_features': 1433,
        'num_classes': 7,
        'num_train': 140,  # Approximate
        'num_test': 1000,
        'num_val': 500
    }