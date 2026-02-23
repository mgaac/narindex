#!/usr/bin/env python3
"""
Neural Turing Machines (NTM) Experiment Runner

Paper: "Neural Turing Machines" (Graves et al., 2014)
       arXiv:1410.5401v2

Usage:
    python experiment.py --num-steps 20000 --sequence-length 10
    python experiment.py --num-steps 50000 --sequence-length 20 --memory-size 128 20
"""

import argparse
import sys
import os
import csv
import random
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tqdm import tqdm

from model import NeuralTuringMachine
from train import loss_fn
from utils.common import setup_reproducibility


# Default network configuration
DEFAULT_CONFIG = {
    'memory_size': [64, 10],
    'hidden_dim': 64,
    'num_shared_layers': 10,
    'num_controller_layers': 10,
    'num_output_layers': 10,
}


def print_experiment_header(config: dict):
    """Print standardized experiment header."""
    print(f"\n{'='*70}")
    print("Neural Turing Machines (NTM)")
    print("Paper: arXiv:1410.5401v2")
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


def analyze_memory_usage(model, memory, write_weights, read_weights):
    """Analyze memory utilization and access patterns."""
    memory_np = np.array(memory)
    write_weights_np = np.array(write_weights)
    read_weights_np = np.array(read_weights)
    
    return {
        'memory_utilization': float(np.mean(np.abs(memory_np))),
        'memory_sparsity': float(np.mean(memory_np == 0)),
        'write_entropy': float(-np.sum(write_weights_np * np.log(write_weights_np + 1e-8))),
        'read_entropy': float(-np.sum(read_weights_np * np.log(read_weights_np + 1e-8))),
        'write_concentration': float(np.max(write_weights_np)),
        'read_concentration': float(np.max(read_weights_np)),
        'active_memory_locations': int(np.sum(np.max(np.abs(memory_np), axis=1) > 0.1))
    }


def copy_task_data_generator(max_sequence_length, element_length):
    """Generate copy task training pairs."""
    while True:
        copy_len = random.randint(1, max_sequence_length - 1)
        target = mx.random.randint(0, 2, shape=[copy_len, element_length])
        input_seq = mx.concatenate([target, mx.ones([1, element_length])])
        input_seq = mx.concatenate([input_seq, mx.zeros([copy_len, element_length])])
        target_seq = mx.concatenate([mx.zeros([copy_len + 1, element_length]), target])
        yield input_seq, target_seq, copy_len


def copy_task_accuracy(model, input_seq, target_seq, copy_len, memory_size):
    """Evaluate copy task accuracy."""
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


def run_experiment(args):
    """Run NTM training experiment."""
    setup_reproducibility()
    
    # Build configuration
    config = DEFAULT_CONFIG.copy()
    if args.memory_size:
        config['memory_size'] = list(args.memory_size)
    if args.hidden_dim:
        config['hidden_dim'] = args.hidden_dim
    
    memory_size = tuple(config['memory_size'])
    
    # Experiment configuration for display
    experiment_config = {
        'Dataset': 'Copy Task (Generated)',
        'Sequence Length': f"1-{args.sequence_length}",
        'Element Length': args.element_length,
        'Memory Size': f"{memory_size[0]} × {memory_size[1]}",
        'Hidden Dim': config['hidden_dim'],
        'Training Steps': args.num_steps,
        'Learning Rate': args.learning_rate,
        'Log Interval': args.log_interval
    }
    
    print_experiment_header(experiment_config)
    
    # Initialize model
    print("Initializing model...")
    model = NeuralTuringMachine(
        input_dim=args.element_length,
        output_dim=args.element_length,
        hidden_dim=config['hidden_dim'],
        num_shared_layers=config['num_shared_layers'],
        num_controller_layers=config['num_controller_layers'],
        num_output_layers=config['num_output_layers'],
        memory_size=memory_size
    )
    mx.eval(model.parameters())
    
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count:,}")
    
    # Initialize optimizer
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    
    # Training functions
    def forward_loss(input_seq, target_seq, r, w, memory, copy_len):
        loss_total = 0
        activations = {}
        
        for i, sequence in enumerate(input_seq):
            logits, r, w, memory, step_activations = model(sequence, r, w, memory)
            activations.update({f"step_{i}_{k}": v for k, v in step_activations.items()})
            if i > copy_len:
                loss_total += loss_fn(logits, target_seq[i])
        
        return loss_total / len(input_seq), r, w, memory, logits, activations
    
    state = [model.state, optimizer.state]
    
    @partial(mx.compile, inputs=state, outputs=state)
    def step(input_seq, target_seq, r, w, memory, copy_len):
        value_and_grad_fn = nn.value_and_grad(model, forward_loss)
        (loss, r, w, memory, logits, activations), grads = value_and_grad_fn(
            input_seq, target_seq, r, w, memory, copy_len
        )
        grads, _ = optim.clip_grad_norm(grads, max_norm=2.0)
        optimizer.update(model, grads)
        return (loss, r, w, memory, logits, activations), grads
    
    # Initialize memory state
    r = mx.ones(memory_size[1]) * 1e-2
    w = mx.ones(memory_size[0]) * 1e-2
    memory = mx.ones(memory_size) * 1e-6
    
    # Training data generator
    train_iter = copy_task_data_generator(args.sequence_length, args.element_length)
    
    # Training tracking
    best_accuracy = 0.0
    best_step = 0
    
    # Create output directory
    os.makedirs('run_log', exist_ok=True)
    log_file = f'run_log/ntm_{args.sequence_length}_{args.num_steps}.csv'
    
    print(f"\nStarting training for {args.num_steps:,} steps...")
    print(f"Logging to: {log_file}")
    
    # Training loop
    with open(log_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "Step", "Loss", "Accuracy", "Memory Utilization", "Active Memory"
        ])
        writer.writeheader()
        
        pbar = tqdm(range(args.num_steps), desc="Training NTM", unit="step")
        
        for step_idx in pbar:
            input_seq, target_seq, copy_len = next(train_iter)
            
            (loss, r, w, memory, logits, activations), grads = step(
                input_seq, target_seq, r, w, memory, copy_len
            )
            mx.eval(model.parameters(), optimizer.state)
            
            if step_idx % args.log_interval == 0:
                accuracy = copy_task_accuracy(model, input_seq, target_seq, copy_len, memory_size)
                mem_analysis = analyze_memory_usage(model, memory, w, r)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_step = step_idx
                
                writer.writerow({
                    "Step": step_idx,
                    "Loss": float(loss),
                    "Accuracy": float(accuracy),
                    "Memory Utilization": mem_analysis['memory_utilization'],
                    "Active Memory": mem_analysis['active_memory_locations']
                })
                csvfile.flush()
                
                pbar.set_postfix({
                    'loss': f"{float(loss):.6f}",
                    'acc': f"{accuracy:.3f}",
                    'best': f"{best_accuracy:.3f}"
                })
                
            else:
                pbar.set_postfix({'loss': f"{float(loss):.6f}"})
    
    # Final evaluation
    final_accuracy = copy_task_accuracy(model, input_seq, target_seq, copy_len, memory_size)
    final_mem_analysis = analyze_memory_usage(model, memory, w, r)
    
    # Print results
    print(f"\n{'='*70}")
    print("Training Complete")
    print(f"{'='*70}")
    print(f"  Final Loss: {float(loss):.6f}")
    print(f"  Final Accuracy: {final_accuracy:.4f}")
    print(f"  Best Accuracy: {best_accuracy:.4f} (step {best_step:,})")
    print(f"  Memory Utilization: {final_mem_analysis['memory_utilization']:.4f}")
    print(f"  Active Memory: {final_mem_analysis['active_memory_locations']}/{memory_size[0]}")
    print(f"{'='*70}\n")
    
    return {
        'final_loss': float(loss),
        'final_accuracy': final_accuracy,
        'best_accuracy': best_accuracy,
        'best_step': best_step
    }


def main():
    parser = argparse.ArgumentParser(
        description="Neural Turing Machines (NTM) Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiment.py --num-steps 20000 --sequence-length 10
  python experiment.py --num-steps 50000 --sequence-length 20 --memory-size 128 20
  python experiment.py --hidden-dim 200 --num-steps 30000
        """
    )
    
    # Training parameters
    parser.add_argument('--num-steps', type=int, default=20000,
                        help='Number of training steps (default: 20000)')
    parser.add_argument('--learning-rate', type=float, default=0.00006,
                        help='Learning rate (default: 0.00006)')
    parser.add_argument('--log-interval', type=int, default=1000,
                        help='Logging interval in steps (default: 1000)')
    
    # Task parameters
    parser.add_argument('--sequence-length', type=int, default=10,
                        help='Maximum sequence length for copy task (default: 10)')
    parser.add_argument('--element-length', type=int, default=5,
                        help='Length of each sequence element (default: 5)')
    
    # Architecture parameters
    parser.add_argument('--memory-size', type=int, nargs=2, metavar=('ROWS', 'COLS'),
                        help='Memory size [rows, cols] (default: [64, 10])')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden dimension (default: 64)')
    
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
