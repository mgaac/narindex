"""
Common utilities for Neural Algorithmic Reasoning implementations.
Provides standardized training, evaluation, and argument parsing patterns.
"""

import argparse
import mlx.core as mx
import mlx.nn as nn
from tqdm import tqdm
from typing import Dict, Tuple


def standard_arg_parser(description: str) -> argparse.ArgumentParser:
    """Create a standardized argument parser with common options."""
    parser = argparse.ArgumentParser(description=description)
    
    # Data arguments
    parser.add_argument(
        "--data", type=str, default="data/",
        help="Path to the dataset directory"
    )
    
    # Training arguments
    parser.add_argument(
        "--learning-rate", type=float, default=3e-3,
        help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--num-steps", type=int, default=10000,
        help="Number of training steps"
    )
    parser.add_argument(
        "--eval-interval", type=int, default=50,
        help="Evaluation interval (steps)"
    )
    
    # Model arguments (can be extended by specific implementations)
    parser.add_argument(
        "--dropout", type=float, default=0.5,
        help="Dropout probability"
    )
    
    return parser


def cross_entropy_loss_fn(model, data, labels, mask):
    """Standard cross-entropy loss function with masking."""
    model = model.train()
    logits = model(data)
    loss = nn.losses.cross_entropy(logits, labels, axis=1) * mask
    n_samples = mask.sum().sum()
    return loss.sum() / n_samples


def accuracy_fn(model, data, labels):
    """Standard accuracy evaluation function."""
    model = model.eval()
    logits = model(data)
    logits = mx.softmax(logits, axis=-1)
    return mx.mean(mx.argmax(logits, axis=-1) == mx.argmax(labels, axis=-1))


class TrainingLoop:
    """Standardized training loop with consistent progress reporting."""
    
    def __init__(self, model, optimizer, loss_fn, eval_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn
        self.loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    def train(self, train_data, test_data, num_steps: int, eval_interval: int = 50):
        """Execute training loop with consistent progress reporting."""
        pbar = tqdm(range(num_steps), desc="Training", unit="steps")
        
        for step in pbar:
            # Training step
            loss, grads = self.loss_and_grad_fn(self.model, *train_data)
            self.optimizer.update(self.model, grads)
            
            # Periodic evaluation
            if step % eval_interval == 0:
                test_loss = self.loss_fn(self.model, *test_data)
                accuracy = self.eval_fn(self.model, test_data[0], test_data[1])
                
                pbar.set_postfix(
                    train_loss=f"{loss:.4f}",
                    test_loss=f"{test_loss:.4f}",
                    accuracy=f"{accuracy:.4f}"
                )
        
        return {"final_loss": float(loss), "final_accuracy": float(accuracy)}


def print_model_summary(model, model_name: str):
    """Print standardized model summary."""
    print(f"\n{model_name} Model Summary")
    print("=" * (len(model_name) + 14))
    print(model)
    
    # Calculate total parameters
    from mlx.utils import tree_flatten
    num_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"\nTotal parameters: {num_params:,}")
    print("-" * (len(model_name) + 14))


def validate_data_shapes(data: Tuple, expected_shapes: Dict[str, Tuple]):
    """Validate input data shapes with descriptive error messages."""
    for i, (name, expected_shape) in enumerate(expected_shapes.items()):
        if i >= len(data):
            raise ValueError(f"Missing data component: {name}")
        
        actual_shape = data[i].shape
        if len(actual_shape) != len(expected_shape):
            raise ValueError(
                f"{name} shape mismatch: expected {len(expected_shape)} dimensions, "
                f"got {len(actual_shape)} dimensions"
            )
        
        # Check specific dimensions (None means any size is acceptable)
        for dim_idx, (expected, actual) in enumerate(zip(expected_shape, actual_shape)):
            if expected is not None and expected != actual:
                raise ValueError(
                    f"{name} dimension {dim_idx} mismatch: expected {expected}, got {actual}"
                )


def setup_reproducibility(seed: int = 42):
    """Setup reproducible random state."""
    mx.random.seed(seed)


# Export commonly used functions
__all__ = [
    'standard_arg_parser',
    'cross_entropy_loss_fn', 
    'accuracy_fn',
    'TrainingLoop',
    'print_model_summary',
    'validate_data_shapes',
    'setup_reproducibility'
]
