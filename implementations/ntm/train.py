"""
Neural Turing Machine (NTM) training script for copy task.
Uses standardized utilities while maintaining specialized NTM functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
from functools import partial

from model import NeuralTuringMachine
from utils.common import standard_arg_parser, setup_reproducibility, print_model_summary
from utils.analysis import compute_gradient_stats, compute_activation_stats, compute_param_stats, compute_update_ratios, TrainingLogger

import random
import argparse
from tqdm import tqdm


def create_ntm_arg_parser():
    """Create argument parser with NTM-specific options."""
    parser = argparse.ArgumentParser(description="Train a Neural Turing Machine on copy task")
    
    # NTM-specific architecture arguments
    parser.add_argument("--input-dim", type=int, default=5, help="Input dimension")
    parser.add_argument("--output-dim", type=int, default=5, help="Output dimension")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--num-shared-layers", type=int, default=10, help="Number of shared layers")
    parser.add_argument("--num-controller-layers", type=int, default=10, help="Number of controller layers")
    parser.add_argument("--num-output-layers", type=int, default=10, help="Number of output layers")
    parser.add_argument("--memory-rows", type=int, default=64, help="Memory matrix rows")
    parser.add_argument("--memory-cols", type=int, default=10, help="Memory matrix columns")
    
    # Training arguments
    parser.add_argument("--learning-rate", type=float, default=6e-5, help="Learning rate")
    parser.add_argument("--num-steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--max-seq-len", type=int, default=10, help="Maximum sequence length")
    parser.add_argument("--log-interval", type=int, default=1000, help="Logging interval")
    
    # Output arguments
    parser.add_argument("--log-file", type=str, default="ntm_training.csv", help="Training log file")
    
    return parser


def init_fn(x, initializer=nn.init.he_normal()):
    """Initialize parameters with He normal initialization."""
    if hasattr(x, 'shape') and len(x.shape) >= 2:
        return initializer(x)
    else:
        return x


def is_leaf(node):
    """Check if node is a leaf in the parameter tree."""
    return isinstance(node, mx.array)


def copy_task_data_generator(max_sequence_length: int, element_length: int):
    """
    Generate copy task data.
    
    Args:
        max_sequence_length: Maximum sequence length
        element_length: Length of each element
        
    Yields:
        Tuple of (input, target, copy_len)
    """
    while True:
        copy_len = random.randint(1, max_sequence_length - 1)
        target = mx.random.randint(0, 2, shape=[copy_len, element_length]).astype(mx.float32)
        
        # Input: [target_sequence, delimiter, zeros_for_output]
        delimiter = mx.ones([1, element_length])
        input_padding = mx.zeros([copy_len, element_length])
        input_seq = mx.concatenate([target, delimiter, input_padding])
        
        # Target: [zeros_for_input, target_sequence] 
        target_padding = mx.zeros([copy_len + 1, element_length])
        target_seq = mx.concatenate([target_padding, target])
        
        yield input_seq, target_seq, copy_len


def binary_cross_entropy_loss(logits, targets):
    """Binary cross-entropy loss for copy task."""
    return nn.losses.binary_cross_entropy(logits, targets)


def copy_task_accuracy(logits, targets):
    """Compute accuracy for copy task."""
    predictions = mx.sigmoid(logits)
    predictions = mx.where(predictions > 0.5, mx.ones_like(predictions), mx.zeros_like(predictions))
    correct = predictions == targets
    return correct.sum() / correct.size


def copy_task_eval_fn(model, input_seq, target_seq, copy_len, memory_size):
    """Evaluate model on copy task."""
    accuracies = []
    read_vector = mx.zeros(memory_size[1])
    write_weights = mx.zeros(memory_size[0])
    memory = mx.ones(memory_size) * 1e-6
    
    for i, input_step in enumerate(input_seq):
        output, read_vector, write_weights, memory, _ = model(
            input_step, read_vector, write_weights, memory
        )
        
        # Only evaluate accuracy after the delimiter
        if i > copy_len:
            acc = copy_task_accuracy(output, target_seq[i])
            accuracies.append(acc)
    
    return mx.array(accuracies).mean() if accuracies else mx.array(0.0)


def forward_loss(model, input_seq, target_seq, memory_size, copy_len):
    """Forward pass with loss computation."""
    total_loss = 0.0
    activations = {}
    read_vector = mx.zeros(memory_size[1])
    write_weights = mx.zeros(memory_size[0])
    memory = mx.ones(memory_size) * 1e-6
    
    for i, input_step in enumerate(input_seq):
        output, read_vector, write_weights, memory, step_activations = model(
            input_step, read_vector, write_weights, memory
        )
        
        # Only compute loss after the delimiter
        if i > copy_len:
            loss = binary_cross_entropy_loss(output, target_seq[i])
            total_loss += loss
        
        # Store activations from last step for analysis
        if i == len(input_seq) - 1:
            activations = step_activations
    
    avg_loss = total_loss / max(1, len(input_seq) - copy_len - 1)
    return avg_loss, read_vector, write_weights, memory, output, activations


def main():
    """Main training function."""
    args = create_ntm_arg_parser().parse_args()
    
    # Setup reproducibility
    setup_reproducibility()
    
    # Model configuration
    memory_size = (args.memory_rows, args.memory_cols)
    model_config = {
        'input_dim': args.input_dim,
        'output_dim': args.output_dim,
        'hidden_dim': args.hidden_dim,
        'num_shared_layers': args.num_shared_layers,
        'num_controller_layers': args.num_controller_layers,
        'num_output_layers': args.num_output_layers,
        'memory_size': memory_size
    }
    
    # Initialize model and optimizer
    print("Initializing Neural Turing Machine...")
    model = NeuralTuringMachine(**model_config)
    
    # Initialize parameters with He normal
    new_params = tree_map(init_fn, model.state, is_leaf=is_leaf)
    model.update(new_params)
    
    optimizer = optim.Adam(args.learning_rate)
    
    # Print model summary
    print_model_summary(model, "Neural Turing Machine")
    
    # Setup training logger
    logger = TrainingLogger(args.log_file)
    
    # Create compiled training step
    state = [model.state, optimizer.state]
    
    @partial(mx.compile, inputs=state, outputs=state)
    def step(input_seq, target_seq, copy_len):
        value_and_grad_fn = nn.value_and_grad(model, lambda m, *args: forward_loss(m, *args)[0])
        (loss, read_vector, write_weights, memory, output, activations), grads = value_and_grad_fn(
            model, input_seq, target_seq, memory_size, copy_len
        )
        grads, _ = optim.clip_grad_norm(grads, max_norm=2.0)
        optimizer.update(model, grads)
        return loss, read_vector, write_weights, memory, output, activations, grads
    
    # Training loop
    print(f"\nStarting training for {args.num_steps} steps...")
    data_generator = copy_task_data_generator(args.max_seq_len, args.input_dim)
    
    # Track previous parameters for update ratios
    prev_params = {k: v.copy() for k, v in model.parameters().items()}
    
    pbar = tqdm(range(args.num_steps), desc="Training NTM", unit="steps")
    
    for step_idx in pbar:
        # Generate training data
        input_seq, target_seq, copy_len = next(data_generator)
        
        # Training step
        loss, read_vector, write_weights, memory, output, activations, grads = step(
            input_seq, target_seq, copy_len
        )
        mx.eval(model.parameters(), optimizer.state)
        
        # Update progress bar
        pbar.set_postfix({"Loss": f"{loss.item():.6f}"})
        
        # Periodic logging and evaluation
        if step_idx % args.log_interval == 0:
            # Compute accuracy
            accuracy = copy_task_eval_fn(model, input_seq, target_seq, copy_len, memory_size)
            
            # Compute detailed statistics
            grad_stats = compute_gradient_stats(grads)
            activation_stats = compute_activation_stats(activations)
            param_stats = compute_param_stats(model.parameters())
            update_ratios = compute_update_ratios(prev_params, model.parameters())
            
            # Log metrics
            metrics = {
                "loss": loss.item(),
                "accuracy": accuracy.item(),
                "learning_rate": args.learning_rate,
                "gradient_stats": grad_stats,
                "activation_stats": activation_stats,
                "parameter_stats": param_stats,
                "update_ratios": update_ratios
            }
            logger.log_step(step_idx, metrics)
            
            # Update previous parameters
            prev_params = {k: v.copy() for k, v in model.parameters().items()}
            
            print(f"\nStep {step_idx}: Loss={loss.item():.6f}, Accuracy={accuracy.item():.4f}")
    
    # Save training logs
    logger.save_logs()
    
    # Final evaluation
    print(f"\nTraining completed!")
    final_input, final_target, final_copy_len = next(data_generator)
    final_accuracy = copy_task_eval_fn(model, final_input, final_target, final_copy_len, memory_size)
    print(f"Final accuracy: {final_accuracy.item():.4f}")
    
    # Print training summary
    summary = logger.get_summary()
    print(f"\nTraining Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")


if __name__ == '__main__':
    main()
