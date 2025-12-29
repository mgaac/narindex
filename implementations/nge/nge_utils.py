"""Utility functions for Neural Graph Execution (NGE) training.

Provides accuracy metrics calculation, loss computation, and debugging utilities.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as utils
import numpy as np


def extract_per_head_magnitude_grads(grads):
    """Extract the L2 norm of gradients for each model component.
    
    Args:
        grads: Gradient tree from value_and_grad
        
    Returns:
        Dictionary mapping component names to gradient magnitudes
    """
    head_names = set()
    utils.tree_map_with_path(lambda path, _: head_names.add(path.split('.')[0]), grads)
    
    per_head_magnitude_grads = {}
    for head_name in head_names:
        per_head_magnitude_grads[head_name] = utils.tree_reduce(
            lambda acc, x: acc + mx.sum(mx.square(x)), grads[head_name], 0.0
        ) ** 0.5
    return per_head_magnitude_grads


def calculate_losses_and_accuracies(model, graph_data, embedding_dim: int):
    """Calculate losses and accuracies for a single graph.
    
    Follows the exact loss computation logic from the training loop.
    
    Args:
        model: NGE model
        graph_data: Dictionary containing graph data and targets
        embedding_dim: Model embedding dimension
        
    Returns:
        Tuple of (aux_losses[5], total_loss, accuracies[5])
    """
    accumulated_loss = mx.array(0.0)
    accumulated_aux_losses = mx.zeros([5])  # [bf_dist, bf_pred, bfs_state, bf_term, bfs_term]
    
    # Accuracy counters - accumulate across all steps
    bf_distance_correct_sum = 0
    bf_distance_total_sum = 0
    bf_predecessor_correct_sum = 0
    bf_predecessor_total_sum = 0
    bfs_state_correct_sum = 0
    bfs_state_total_sum = 0
    bf_termination_correct_sum = 0
    bf_termination_total_sum = 0
    bfs_termination_correct_sum = 0
    bfs_termination_total_sum = 0
    
    num_nodes = graph_data['num_nodes']
    previous_step_hidden_states = mx.zeros([num_nodes, 2 * embedding_dim])
    
    num_bf_steps = len(graph_data['bf_distance_targets'])
    num_bfs_steps = len(graph_data['bfs_state_targets'])
    num_steps = max(num_bf_steps, num_bfs_steps)
    
    for i in range(num_steps):
        # Check if samples exist
        bf_sample_exists = (i + 1) < num_bf_steps
        bfs_sample_exists = (i + 1) < num_bfs_steps
        
        if not (bf_sample_exists or bfs_sample_exists):
            continue
        
        # Prepare data
        if bfs_sample_exists:
            true_bfs_state = graph_data['bfs_state_targets'][i]
            target_bfs_state = graph_data['bfs_state_targets'][i + 1]
        else:
            true_bfs_state = graph_data['bfs_state_targets'][-1]
            target_bfs_state = graph_data['bfs_state_targets'][-1]
        
        if bf_sample_exists:
            true_distance_bf = graph_data['bf_distance_targets'][i]
            target_distance_bf = graph_data['bf_distance_targets'][i + 1]
            target_predecessor_bf = graph_data['bf_predecessor_targets'][i + 1]
        else:
            true_distance_bf = graph_data['bf_distance_targets'][-1]
            target_distance_bf = graph_data['bf_distance_targets'][-1]
            target_predecessor_bf = graph_data['bf_predecessor_targets'][-1]
        
        # Generate termination targets
        is_last_bf_step = (i + 1) == (num_bf_steps - 1)
        is_last_bfs_step = (i + 1) == (num_bfs_steps - 1)
        termination_targets = {
            'bf': mx.array(1.0 if is_last_bf_step else 0.0),
            'bfs': mx.array(1.0 if is_last_bfs_step else 0.0)
        }
        
        # Prepare model inputs
        node_algo_features = mx.stack([true_bfs_state, true_distance_bf], axis=1)
        input_embeddings = mx.concatenate([previous_step_hidden_states, node_algo_features], axis=1)
        model_input = (input_embeddings, graph_data['edge_matrix'])
        
        # Forward pass
        bfs_output, bf_output, termination_probs, processed_embeddings = model(model_input)
        
        # Compute losses and accuracies
        if bf_sample_exists:
            bf_distance_predictions, bf_predecessor_predictions = bf_output
            
            # BF Distance Loss
            bf_distance_loss = nn.losses.mse_loss(
                bf_distance_predictions, target_distance_bf, reduction='mean'
            )
            
            # BF Distance Accuracy
            distance_errors = mx.abs(bf_distance_predictions - target_distance_bf)
            distance_tolerance = 0.1  # 10% tolerance on normalized scale
            distance_correct = mx.sum(distance_errors <= distance_tolerance).item()
            bf_distance_correct_sum += distance_correct
            bf_distance_total_sum += num_nodes
            
            # BF Predecessor Loss
            valid_mask = (target_predecessor_bf != -1)
            safe_targets = mx.where(valid_mask, target_predecessor_bf, mx.zeros_like(target_predecessor_bf))
            
            # Only compute loss if we have valid targets
            if mx.sum(valid_mask) > 0:
                per_node_ce = nn.losses.cross_entropy(
                    bf_predecessor_predictions, safe_targets, reduction='none', label_smoothing=1e-6
                )
                valid_mask_f = valid_mask.astype(mx.float32)
                denom = mx.maximum(valid_mask_f.sum(), mx.array(1.0))
                bf_predecessor_loss = (per_node_ce * valid_mask_f).sum() / denom
            else:
                bf_predecessor_loss = mx.array(0.0)
            
            # BF Predecessor Accuracy - only count valid nodes
            pred_argmax = mx.argmax(bf_predecessor_predictions, axis=-1)
            pred_correct = mx.sum((pred_argmax == target_predecessor_bf) & valid_mask).item()
            pred_total = mx.sum(valid_mask).item()
            bf_predecessor_correct_sum += pred_correct
            bf_predecessor_total_sum += pred_total
            
            # BF Termination Loss
            bf_termination_loss = nn.losses.binary_cross_entropy(
                termination_probs['bf'], termination_targets['bf'], reduction='mean', with_logits=True
            )
            
            # BF Termination Accuracy
            bf_term_prob = mx.sigmoid(termination_probs['bf'])
            bf_term_pred = (bf_term_prob > 0.5).astype(mx.float32)
            bf_term_correct = (bf_term_pred == termination_targets['bf']).item()
            bf_termination_correct_sum += bf_term_correct
            bf_termination_total_sum += 1
        else:
            bf_distance_loss = mx.array(0.0)
            bf_predecessor_loss = mx.array(0.0)
            bf_termination_loss = mx.array(0.0)
        
        if bfs_sample_exists:
            # BFS State Loss
            bfs_state_loss = nn.losses.binary_cross_entropy(
                bfs_output, target_bfs_state, reduction='mean', with_logits=True
            )
            
            # BFS State Accuracy
            bfs_state_probs = mx.sigmoid(bfs_output)
            bfs_state_pred = (bfs_state_probs > 0.5).astype(mx.float32)
            bfs_correct = mx.sum(bfs_state_pred == target_bfs_state).item()
            bfs_state_correct_sum += bfs_correct
            bfs_state_total_sum += num_nodes
            
            # BFS Termination Loss
            bfs_termination_loss = nn.losses.binary_cross_entropy(
                termination_probs['bfs'], termination_targets['bfs'], reduction='mean', with_logits=True
            )
            
            # BFS Termination Accuracy
            bfs_term_prob = mx.sigmoid(termination_probs['bfs'])
            bfs_term_pred = (bfs_term_prob > 0.5).astype(mx.float32)
            bfs_term_correct = (bfs_term_pred == termination_targets['bfs']).item()
            bfs_termination_correct_sum += bfs_term_correct
            bfs_termination_total_sum += 1
        else:
            bfs_state_loss = mx.array(0.0)
            bfs_termination_loss = mx.array(0.0)
        
        # Accumulate losses
        raw_losses = mx.array([
            bf_distance_loss, bf_predecessor_loss, bfs_state_loss,
            bf_termination_loss, bfs_termination_loss
        ])
        total_step_loss = mx.sum(raw_losses)
        accumulated_loss += total_step_loss
        accumulated_aux_losses += raw_losses
        
        # Update hidden states for next step
        previous_step_hidden_states = processed_embeddings
    
    # Calculate averages
    bf_steps = max(num_bf_steps - 1, 0)
    bfs_steps = max(num_bfs_steps - 1, 0)
    effective_steps = max(bf_steps, bfs_steps, 1)
    
    average_loss = accumulated_loss / effective_steps
    
    # Per-task averaging
    per_task_counter = mx.array([
        max(bf_steps, 1),   # bf_distance
        max(bf_steps, 1),   # bf_predecessor
        max(bfs_steps, 1),  # bfs_state
        max(bf_steps, 1),   # bf_termination
        max(bfs_steps, 1),  # bfs_termination
    ], dtype=mx.float32)
    avg_aux_losses = accumulated_aux_losses / per_task_counter
    
    # Calculate overall accuracies
    bf_distance_acc = bf_distance_correct_sum / max(bf_distance_total_sum, 1)
    bf_predecessor_acc = bf_predecessor_correct_sum / max(bf_predecessor_total_sum, 1)
    bfs_state_acc = bfs_state_correct_sum / max(bfs_state_total_sum, 1)
    bf_termination_acc = bf_termination_correct_sum / max(bf_termination_total_sum, 1)
    bfs_termination_acc = bfs_termination_correct_sum / max(bfs_termination_total_sum, 1)
    
    accuracies = mx.array([
        bf_distance_acc, bf_predecessor_acc, bfs_state_acc,
        bf_termination_acc, bfs_termination_acc
    ])
    
    return avg_aux_losses, average_loss, accuracies


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


def print_model_info(model):
    """Print basic model information."""
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count:,}")
