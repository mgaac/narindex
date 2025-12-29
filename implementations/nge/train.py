"""Neural Execution of Graph Algorithms (NGE) Training Module

Provides the training loop and loss computation for the NGE model
on Bellman-Ford and BFS algorithm execution tasks.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as utils
import mlx.optimizers as optim

from model import NGE, AggregationFunction
from nge_utils import extract_per_head_magnitude_grads, calculate_losses_and_accuracies


def graph_execution_loss_fn(model, graph_data, embedding_dim: int, bf_pred_alpha: float = 1.0, label_smoothing: float = 0.0):
    """Compute loss for a single graph across all algorithm execution steps.
    
    Args:
        model: NGE model
        graph_data: Dictionary containing graph data and targets
        embedding_dim: Model embedding dimension
        bf_pred_alpha: Weight for BF predecessor loss
        label_smoothing: Label smoothing value for cross-entropy
        
    Returns:
        Tuple of (average_loss, avg_aux_losses[5])
    """
    accumulated_loss = mx.array(0.0)
    accumulated_aux_losses = mx.zeros([5])
    
    num_nodes = graph_data['num_nodes']
    previous_step_hidden_states = mx.zeros([num_nodes, embedding_dim * 2])
    
    num_bf_steps = len(graph_data['bf_distance_targets'])
    num_bfs_steps = len(graph_data['bfs_state_targets'])
    
    num_steps = max(num_bf_steps, num_bfs_steps)
    
    for i in range(num_steps):
        # Check if samples exist
        bf_sample_exists = (i + 1) < num_bf_steps
        bfs_sample_exists = (i + 1) < num_bfs_steps
        
        if not (bf_sample_exists or bfs_sample_exists):
            continue
        
        # Prepare data for current step
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
        
        # Compute losses
        if bf_sample_exists:
            bf_distance_predictions, bf_predecessor_predictions = bf_output
            bf_distance_loss = nn.losses.mse_loss(
                bf_distance_predictions, target_distance_bf, reduction='mean'
            )
            
            # Convert invalid, denoted by -1, to a valid class, 0
            valid_mask = (target_predecessor_bf != -1)
            safe_targets = mx.where(valid_mask, target_predecessor_bf, mx.zeros_like(target_predecessor_bf))
            
            per_node_ce = nn.losses.cross_entropy(
                bf_predecessor_predictions, safe_targets, reduction='none', label_smoothing=label_smoothing
            )
            
            # Only consider loss over valid nodes
            valid_mask_f = valid_mask.astype(mx.float32)
            denom = mx.maximum(valid_mask_f.sum(), mx.array(1.0))
            bf_predecessor_loss = (per_node_ce * valid_mask_f).sum() / denom
            
            bf_termination_loss = nn.losses.binary_cross_entropy(
                termination_probs['bf'], termination_targets['bf'], reduction='mean', with_logits=True
            )
        else:
            bf_distance_loss = mx.array(0.0)
            bf_predecessor_loss = mx.array(0.0)
            bf_termination_loss = mx.array(0.0)
        
        if bfs_sample_exists:
            bfs_state_loss = nn.losses.binary_cross_entropy(
                bfs_output, target_bfs_state, reduction='mean', with_logits=True
            )
            bfs_termination_loss = nn.losses.binary_cross_entropy(
                termination_probs['bfs'], termination_targets['bfs'], reduction='mean', with_logits=True
            )
        else:
            bfs_state_loss = mx.array(0.0)
            bfs_termination_loss = mx.array(0.0)
        
        raw_losses = mx.array([
            bf_distance_loss, bf_pred_alpha * bf_predecessor_loss, bfs_state_loss,
            bf_termination_loss, bfs_termination_loss
        ])
        total_step_loss = mx.sum(raw_losses)
        
        # Update for next step
        previous_step_hidden_states = processed_embeddings
        accumulated_loss += total_step_loss
        accumulated_aux_losses += raw_losses
    
    # Compute averages AFTER the loop
    bf_steps = max(num_bf_steps - 1, 0)
    bfs_steps = max(num_bfs_steps - 1, 0)
    effective_steps = max(bf_steps, bfs_steps, 1)
    
    average_loss = accumulated_loss / effective_steps
    per_task_counter = mx.array([
        max(bf_steps, 1),   # bf_distance
        max(bf_steps, 1),   # bf_predecessor
        max(bfs_steps, 1),  # bfs_state
        max(bf_steps, 1),   # bf_termination
        max(bfs_steps, 1),  # bfs_termination
    ], dtype=mx.float32)
    avg_aux_losses = accumulated_aux_losses / per_task_counter
    
    return average_loss, avg_aux_losses


def evaluate_model(model, dataset, embedding_dim: int):
    """Evaluate model on a dataset.
    
    Args:
        model: NGE model
        dataset: List of graph data dictionaries
        embedding_dim: Model embedding dimension
        
    Returns:
        Tuple of (avg_aux_losses, avg_epoch_loss, avg_accuracies)
    """
    # Set model to evaluation mode (disables dropout)
    model.eval()
    
    accumulated_epoch_loss = mx.array(0.0)
    accumulated_aux_losses = mx.zeros([5])
    accumulated_accuracies = mx.zeros([5])
    
    for graph_data in dataset:
        aux_losses, loss, accuracies = calculate_losses_and_accuracies(model, graph_data, embedding_dim)
        
        accumulated_epoch_loss += loss
        accumulated_aux_losses += aux_losses
        accumulated_accuracies += accuracies
    
    avg_epoch_loss = accumulated_epoch_loss / len(dataset)
    avg_aux_losses = accumulated_aux_losses / len(dataset)
    avg_accuracies = accumulated_accuracies / len(dataset)
    
    # Set model back to training mode
    model.train()
    
    return avg_aux_losses, avg_epoch_loss, avg_accuracies


def train_epoch(
    model,
    optimizer,
    dataset,
    embedding_dim: int,
    batch_size: int = 1,
    max_grad_norm: float = 1.0,
    bf_pred_alpha: float = 1.0,
    label_smoothing: float = 0.0
):
    """Train for one epoch.
    
    Args:
        model: NGE model
        optimizer: Optimizer instance
        dataset: List of graph data dictionaries
        embedding_dim: Model embedding dimension
        batch_size: Gradient accumulation batch size
        max_grad_norm: Maximum gradient norm for clipping
        bf_pred_alpha: Weight for BF predecessor loss
        label_smoothing: Label smoothing value
        
    Returns:
        Tuple of (avg_epoch_loss, avg_aux_losses, avg_per_head_grads)
    """
    model.train()
    
    def loss_fn(model, graph_data):
        return graph_execution_loss_fn(
            model, graph_data, embedding_dim, bf_pred_alpha, label_smoothing
        )
    
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    accumulated_epoch_loss = mx.array(0.0)
    accumulated_aux_losses = mx.zeros([5])
    accumulated_per_head_grads = {}
    
    permutation = mx.random.permutation(len(dataset))
    
    acc_batch_grads = None
    bucket_count = 0
    
    for idx_in_epoch, idx in enumerate(permutation):
        graph_data = dataset[int(idx.item())]
        
        (loss, aux_losses), grads = loss_and_grad_fn(model, graph_data)
        
        per_head_magnitude_grads = extract_per_head_magnitude_grads(grads)
        
        # Accumulate per-head gradients
        for head_name, grad_value in per_head_magnitude_grads.items():
            if head_name not in accumulated_per_head_grads:
                accumulated_per_head_grads[head_name] = grad_value
            else:
                accumulated_per_head_grads[head_name] += grad_value
        
        # Gradient accumulation
        if acc_batch_grads is None:
            acc_batch_grads = grads
        else:
            acc_batch_grads = utils.tree_map(lambda a, b: a + b, acc_batch_grads, grads)
        bucket_count += 1
        
        end_of_bucket = (bucket_count == batch_size)
        end_of_epoch = (idx_in_epoch + 1 == len(permutation))
        if end_of_bucket or end_of_epoch:
            # Average by actual bucket size
            avg_grads = utils.tree_map(lambda x: x / bucket_count, acc_batch_grads)
            
            # Clip gradients
            avg_grads, _ = optim.clip_grad_norm(avg_grads, max_norm=max_grad_norm)
            
            optimizer.update(model, avg_grads)
            mx.eval(model.parameters(), optimizer.state)
            
            # Reset for next bucket
            acc_batch_grads = None
            bucket_count = 0
        
        # Book-keeping for epoch metrics
        accumulated_epoch_loss += loss
        accumulated_aux_losses += aux_losses
    
    avg_epoch_loss = accumulated_epoch_loss / len(dataset)
    avg_aux_losses = accumulated_aux_losses / len(dataset)
    
    # Compute average per-head gradients
    avg_per_head_grads = {
        head_name: grad_value / len(dataset)
        for head_name, grad_value in accumulated_per_head_grads.items()
    }
    
    return avg_epoch_loss, avg_aux_losses, avg_per_head_grads
