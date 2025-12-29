"""
Neural Turing Machine (NTM) Training Module

Provides training utilities for the NTM model on copy task.
For full experiments, use experiment.py instead.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import mlx.core as mx
import mlx.nn as nn

from model import NeuralTuringMachine
from utils.common import setup_reproducibility, print_model_summary


# Default network configuration
network_param = {
    'memory_size': [64, 10],
    'hdim': 64,
    'numl_shared': 10,
    'numl_con': 10,
    'numl_out': 10,
}

train_param = {
    'learning_rate': 6e-5,
    'num_steps': 10000,
    'log_interval': 1000
}


def loss_fn(logits: mx.array, targets: mx.array) -> mx.array:
    """Binary cross-entropy loss for copy task."""
    return nn.losses.binary_cross_entropy(logits, targets)


def copy_task_eval_fn(
    model: NeuralTuringMachine,
    input_seq: mx.array, 
    target_seq: mx.array, 
    copy_len: int, 
    memory_size: tuple
) -> float:
    """Evaluate model on copy task."""
    r = mx.ones(memory_size[1]) * 1e-2
    w = mx.ones(memory_size[0]) * 1e-2
    memory = mx.ones(memory_size) * 1e-6
    
    correct = 0
    total = 0
    
    for i, sequence in enumerate(input_seq):
        logits, r, w, memory, _ = model(sequence, r, w, memory)
        if i > copy_len:
            predictions = mx.sigmoid(logits) > 0.5
            targets = target_seq[i] > 0.5
            correct += int((predictions == targets).sum())
            total += predictions.size
    
    return correct / max(total, 1)


if __name__ == "__main__":
    print("Use experiment.py for full experiment runs.")
    print("This module provides loss_fn() and copy_task_eval_fn() for programmatic use.")
