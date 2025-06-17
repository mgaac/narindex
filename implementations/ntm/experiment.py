#!/usr/bin/env python3
"""
Neural Turing Machines (NTM) Experiment Runner
arXiv:1410.5401v2

Usage:
    python experiment.py --iterations 20000 --sequence-length 10
    python experiment.py --iterations 50000 --sequence-length 20 --memory-size 128 20
"""

import argparse
import os
import sys
import csv
import json
from pathlib import Path

# Add the root directory to Python path for utils import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from tqdm import tqdm
import numpy as np
import random
from functools import partial

from model import controller
from train import network_param, train_param, loss_fn, copy_task_eval_fn
import utils


def print_experiment_header(**config):
    """Print NTM experiment header with configuration."""
    print(f"\n{'='*70}")
    print(f"Neural Turing Machines (NTM) Experiment")
    print(f"Paper: arXiv:1410.5401v2")
    print(f"{'='*70}")
    for key, value in config.items():
        print(f"{key}: {value}")
    print(f"{'='*70}")


def analyze_memory_usage(model, memory, write_weights, read_weights, step, analysis_frequency=1000):
    """Analyze memory utilization and access patterns."""
    if step % analysis_frequency != 0:
        return {}
    
    # Convert to numpy for analysis
    memory_np = np.array(memory)
    write_weights_np = np.array(write_weights)
    read_weights_np = np.array(read_weights)
    
    analysis = {
        'memory_utilization': float(np.mean(np.abs(memory_np))),
        'memory_sparsity': float(np.mean(memory_np == 0)),
        'write_entropy': float(-np.sum(write_weights_np * np.log(write_weights_np + 1e-8))),
        'read_entropy': float(-np.sum(read_weights_np * np.log(read_weights_np + 1e-8))),
        'write_concentration': float(np.max(write_weights_np)),
        'read_concentration': float(np.max(read_weights_np)),
        'active_memory_locations': int(np.sum(np.max(np.abs(memory_np), axis=1) > 0.1))
    }
    
    return analysis


def train_pair_iter(max_sequence_length, element_length):
    """Generate training pairs for copy task."""
    while True:
        copy_len = random.randint(1, max_sequence_length - 1)
        target = mx.random.randint(0, 2, shape=[copy_len, element_length])
        input_seq = mx.concatenate([target, mx.ones([1, element_length])])
        input_seq = mx.concatenate([input_seq, mx.zeros([copy_len, element_length])])
        target_seq = mx.concatenate([mx.zeros([copy_len + 1, element_length]), target])
        yield input_seq, target_seq, copy_len


def run_experiment(args):
    """Run NTM training experiment with memory analysis."""
    
    # Update network parameters based on arguments
    network_param_local = network_param.copy()
    if args.memory_size:
        if len(args.memory_size) == 2:
            network_param_local["memory_size"] = args.memory_size
        else:
            print("Memory size must be specified as two integers: [num_locations, memory_width]")
            return None
    
    if args.controller_size:
        network_param_local["hdim"] = args.controller_size
    
    # Configuration
    experiment_config = {
        'iterations': args.iterations,
        'learning_rate': args.learning_rate,
        'sequence_length': f"1-{args.sequence_length}",
        'element_length': args.element_length,
        'dataset': 'Copy Task (Generated)',
        'memory_size': f"{network_param_local['memory_size'][0]}×{network_param_local['memory_size'][1]}",
        'controller_size': network_param_local['hdim'],
        'memory_locations': network_param_local['memory_size'][0],
        'memory_width': network_param_local['memory_size'][1]
    }
    
    print_experiment_header(**experiment_config)
    
    # Initialize model
    model = controller(
        idim=args.element_length,
        odim=args.element_length,
        hdim=network_param_local["hdim"],
        numl_shared=network_param_local["numl_shared"],
        numl_con=network_param_local["numl_con"],
        numl_out=network_param_local["numl_out"],
        memory_size=network_param_local["memory_size"]
    )
    
    # Initialize optimizer
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    
    # Count parameters correctly for MLX
    def count_parameters(params):
        total = 0
        if isinstance(params, dict):
            for v in params.values():
                if hasattr(v, 'size'):
                    total += v.size
                elif isinstance(v, dict):
                    total += count_parameters(v)
        return total
    
    param_count = count_parameters(model.parameters())
    print(f"Model initialized:")
    print(f"  Memory: {network_param_local['memory_size'][0]} locations × {network_param_local['memory_size'][1]} width")
    print(f"  Controller: {network_param_local['hdim']} hidden units")
    print(f"  Parameters: {param_count:,}")
    
    # Training functions
    def forward_loss(input_seq, target_seq, r, w, memory, copy_len):
        loss_total = 0
        activations = {}
        
        for i, sequence in enumerate(input_seq):
            logits, r, w, memory, step_activations = model(sequence, r, w, memory)
            activations.update({f"step_{i}_{k}": v for k, v in step_activations.items()})
            
            if i > copy_len:  # Only compute loss after delimiter
                loss_total += loss_fn(logits, target_seq[i])
        
        loss = loss_total / len(input_seq)
        return loss, r, w, memory, logits, activations
    
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
    
    # Initialize memory
    r = mx.ones(network_param_local["memory_size"][1]) * 1e-2
    w = mx.ones(network_param_local["memory_size"][0]) * 1e-2
    memory = mx.ones(network_param_local["memory_size"]) * 1e-6
    
    # Training data generator
    train_iter = train_pair_iter(args.sequence_length, args.element_length)
    
    # Training tracking
    best_accuracy = 0.0
    best_iteration = 0
    train_losses = []
    accuracies = []
    memory_analyses = []
    
    # Previous values for delta calculation
    prev_loss = None
    prev_accuracy = None
    
    # Create output directory for detailed logs
    os.makedirs('run_log', exist_ok=True)
    csv_file = f'run_log/experiment_{args.sequence_length}_{args.iterations}.csv'
    
    print(f"\nStarting training for {args.iterations:,} iterations...")
    print(f"Sequence length: 1-{args.sequence_length}, Element length: {args.element_length}")
    print(f"Detailed logs: {csv_file}")
    
    # Training loop with memory analysis
    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = [
            "Iteration", "Training Loss", "Accuracy Score", "Learning Rate",
            "Memory Utilization", "Write Entropy", "Read Entropy", 
            "Active Memory Locations", "Gradient Stats", "Activation Stats",
            "Parameter Stats", "Weight Update Ratio"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Previous parameters for update ratio computation
        prev_params = {k: v.copy() for k, v in model.parameters().items()}
        
        with tqdm(range(args.iterations), desc="Training NTM", unit="iter",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            
            for i in range(args.iterations):
                # Generate training pair
                input_seq, target_seq, copy_len = next(train_iter)
                
                # Training step
                (loss, r, w, memory, logits, activations), grads = step(
                    input_seq, target_seq, r, w, memory, copy_len
                )
                mx.eval(model.parameters(), optimizer.state)
                
                train_losses.append(float(loss))
                
                # Detailed evaluation and logging
                if i % args.log_frequency == 0:
                    # Calculate accuracy
                    acc_value = copy_task_eval_fn(input_seq, target_seq, copy_len)
                    accuracies.append(float(acc_value))
                    
                    # Calculate deltas for logging iterations
                    loss_delta = 0.0 if prev_loss is None else float(loss) - prev_loss
                    acc_delta = 0.0 if prev_accuracy is None else float(acc_value) - prev_accuracy
                    
                    # Track best performance
                    if acc_value > best_accuracy:
                        best_accuracy = acc_value
                        best_iteration = i
                    
                    # Memory analysis
                    memory_analysis = analyze_memory_usage(model, memory, w, r, i, args.log_frequency)
                    if memory_analysis:
                        memory_analyses.append(memory_analysis)
                    
                    # Compute additional statistics
                    grad_stats = utils.compute_gradient_stats(grads)
                    act_stats = utils.compute_activation_stats(activations)
                    param_stats = utils.compute_param_stats(model.parameters())
                    update_ratios = utils.compute_update_ratios(prev_params, model.parameters())
                    
                    # Update previous parameters
                    prev_params = {k: v.copy() for k, v in model.parameters().items()}
                    
                    # Log to CSV
                    log_data = {
                        "Iteration": i,
                        "Training Loss": float(loss),
                        "Accuracy Score": float(acc_value),
                        "Learning Rate": args.learning_rate,
                        "Memory Utilization": memory_analysis.get('memory_utilization', 0),
                        "Write Entropy": memory_analysis.get('write_entropy', 0),
                        "Read Entropy": memory_analysis.get('read_entropy', 0),
                        "Active Memory Locations": memory_analysis.get('active_memory_locations', 0),
                        "Gradient Stats": json.dumps(grad_stats),
                        "Activation Stats": json.dumps(act_stats),
                        "Parameter Stats": json.dumps(param_stats),
                        "Weight Update Ratio": json.dumps(update_ratios)
                    }
                    writer.writerow(log_data)
                    csvfile.flush()
                    
                    # Update progress bar with detailed info including deltas
                    postfix_dict = {
                        'loss': f"{float(loss):.6f}",
                        'accuracy': f"{float(acc_value):.3f}",
                        'best_acc': f"{float(best_accuracy):.3f}",
                        'mem_util': f"{memory_analysis.get('memory_utilization', 0):.3f}",
                        'active_mem': f"{memory_analysis.get('active_memory_locations', 0)}"
                    }
                    
                    if prev_loss is not None:
                        postfix_dict['Δloss'] = f"{loss_delta:+.6f}"
                        postfix_dict['Δacc'] = f"{acc_delta:+.3f}"
                    
                    pbar.set_postfix(postfix_dict)
                    
                    # Update previous values for next delta calculation
                    prev_loss = float(loss)
                    prev_accuracy = float(acc_value)
                else:
                    # Quick update
                    pbar.set_postfix({
                        'loss': f"{float(loss):.6f}",
                        'best_acc': f"{float(best_accuracy):.3f}"
                    })
                
                pbar.update(1)
    
    # Final comprehensive evaluation
    final_loss = float(loss)
    final_accuracy = float(copy_task_eval_fn(input_seq, target_seq, copy_len))
    final_memory_analysis = analyze_memory_usage(model, memory, w, r, args.iterations, 1)
    
    print(f"\n{'='*70}")
    print(f"NTM Training Completed!")
    print(f"{'='*70}")
    print(f"Final Results:")
    print(f"  Final Loss: {final_loss:.6f}")
    print(f"  Final Accuracy: {final_accuracy:.4f}")
    print(f"  Best Accuracy: {float(best_accuracy):.4f} (iteration {best_iteration:,})")
    print(f"  Total Iterations: {args.iterations:,}")
    
    print(f"\nMemory Analysis:")
    print(f"  Memory Utilization: {final_memory_analysis.get('memory_utilization', 0):.4f}")
    print(f"  Active Memory Locations: {final_memory_analysis.get('active_memory_locations', 0)}/{network_param_local['memory_size'][0]}")
    print(f"  Write Attention Entropy: {final_memory_analysis.get('write_entropy', 0):.4f}")
    print(f"  Read Attention Entropy: {final_memory_analysis.get('read_entropy', 0):.4f}")
    print(f"  Write Concentration: {final_memory_analysis.get('write_concentration', 0):.4f}")
    print(f"  Read Concentration: {final_memory_analysis.get('read_concentration', 0):.4f}")
    
    # Training dynamics analysis
    if len(train_losses) > 1000:
        recent_loss_trend = np.mean(train_losses[-500:]) - np.mean(train_losses[-1000:-500])
        print(f"\nTraining Dynamics:")
        print(f"  Recent loss trend: {recent_loss_trend:+.8f}")
        print(f"  Convergence: {'Stable' if abs(recent_loss_trend) < 1e-6 else 'Still improving'}")
    
    # Copy task specific analysis
    print(f"\nCopy Task Performance:")
    print(f"  Task: Learn to copy sequences of length 1-{args.sequence_length}")
    print(f"  Final test sequence length: {copy_len}")
    print(f"  Memory efficiency: {final_memory_analysis.get('active_memory_locations', 0)/copy_len:.2f}x sequence length")
    
    return {
        'final_loss': final_loss,
        'final_accuracy': final_accuracy,
        'best_accuracy': float(best_accuracy),
        'best_iteration': best_iteration,
        'train_losses': train_losses,
        'accuracies': accuracies,
        'memory_analyses': memory_analyses,
        'final_memory_analysis': final_memory_analysis,
        'network_config': network_param_local
    }


def main():
    parser = argparse.ArgumentParser(
        description="Neural Turing Machines (NTM) Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard copy task training
  python experiment.py --iterations 20000 --sequence-length 10
  
  # Longer sequences with more memory
  python experiment.py --iterations 50000 --sequence-length 20 --memory-size 128 20
  
  # Different controller sizes
  python experiment.py --controller-size 200 --iterations 30000
  
  # High-frequency logging for analysis
  python experiment.py --log-frequency 500 --iterations 10000
  
NTM Architecture Details:
  - Neural Turing Machine with external memory
  - Copy task: Learn to store and recall arbitrary sequences
  - Memory attention mechanisms (read/write heads)
  - Content-based and location-based addressing
  
Copy Task:
  - Input: Random binary sequence + delimiter + zeros
  - Output: zeros + original sequence 
  - Tests: Memory storage, retrieval, and sequence manipulation
  
Memory Analysis:
  - Memory utilization: How much of memory is actively used
  - Attention entropy: Diversity of memory access patterns
  - Active locations: Number of memory slots being used
        """
    )
    
    # Training parameters
    parser.add_argument('--iterations', type=int, default=20000,
                        help='Number of training iterations (default: 20000)')
    parser.add_argument('--learning-rate', type=float, default=0.00006,
                        help='Learning rate (default: 0.00006, optimized for NTM)')
    parser.add_argument('--log-frequency', type=int, default=1000,
                        help='Log detailed metrics every N iterations (default: 1000)')
    
    # Copy task parameters
    parser.add_argument('--sequence-length', type=int, default=10,
                        help='Maximum sequence length for copy task (default: 10)')
    parser.add_argument('--element-length', type=int, default=5,
                        help='Length of each sequence element (default: 5)')
    
    # NTM architecture parameters
    parser.add_argument('--memory-size', type=int, nargs=2, metavar=('LOCATIONS', 'WIDTH'),
                        help='Memory size: [num_locations, memory_width] (default: [128, 20])')
    parser.add_argument('--controller-size', type=int,
                        help='Controller hidden size (default: 100)')
    
    args = parser.parse_args()
    
    try:
        results = run_experiment(args)
        if results is None:
            return 1
        return 0
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 