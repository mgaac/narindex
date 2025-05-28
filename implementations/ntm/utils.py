"""
Utility functions for NTM training analysis.
"""

import mlx.core as mx


def compute_gradient_stats(grads):
    """Compute statistics about gradients."""
    grad_norms = []
    grad_means = []
    
    def collect_stats(leaf):
        if hasattr(leaf, 'shape') and len(leaf.shape) > 0:
            grad_norms.append(float(mx.linalg.norm(leaf).item()))
            grad_means.append(float(mx.mean(leaf).item()))
        return leaf
    
    # Use tree_map to traverse the gradient tree
    from mlx.utils import tree_map
    tree_map(collect_stats, grads)
    
    return {
        'grad_norm_mean': sum(grad_norms) / len(grad_norms) if grad_norms else 0.0,
        'grad_norm_max': max(grad_norms) if grad_norms else 0.0,
        'grad_mean': sum(grad_means) / len(grad_means) if grad_means else 0.0
    }


def compute_activation_stats(activations):
    """Compute statistics about activations."""
    if not activations:
        return {}
    
    stats = {}
    for key, value in activations.items():
        if hasattr(value, 'shape') and len(value.shape) > 0:
            stats[f'{key}_mean'] = float(mx.mean(value).item())
            stats[f'{key}_std'] = float(mx.std(value).item())
    
    return stats


def compute_param_stats(parameters):
    """Compute statistics about model parameters."""
    param_norms = []
    param_means = []
    
    for name, param in parameters.items():
        if hasattr(param, 'shape') and len(param.shape) > 0:
            param_norms.append(float(mx.linalg.norm(param).item()))
            param_means.append(float(mx.mean(param).item()))
    
    return {
        'param_norm_mean': sum(param_norms) / len(param_norms) if param_norms else 0.0,
        'param_norm_max': max(param_norms) if param_norms else 0.0,
        'param_mean': sum(param_means) / len(param_means) if param_means else 0.0
    }


def compute_update_ratios(prev_params, current_params):
    """Compute ratios of parameter updates to parameter values."""
    update_ratios = {}
    
    for name in prev_params:
        if name in current_params:
            prev = prev_params[name]
            current = current_params[name]
            
            if hasattr(prev, 'shape') and hasattr(current, 'shape'):
                update = current - prev
                update_norm = mx.linalg.norm(update)
                param_norm = mx.linalg.norm(current)
                
                ratio = update_norm / (param_norm + 1e-8)
                update_ratios[name] = float(ratio.item())
    
    return update_ratios 