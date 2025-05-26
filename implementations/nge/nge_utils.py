import mlx.core as mx
import mlx.nn as nn
import numpy as np
from enum import Enum

# Task enumeration (needed for logging)
class task(Enum):
    PARALLEL_ALGORITHM=0
    SEQUENTIAL_ALGORITHM=1

def calculate_accuracy_metrics(state, predesecor, reachability_target, predesecor_target, 
                             termination_prob, termination_target, distance=None, distance_target=None):
    """Calculate accuracy metrics for all task components"""
    metrics = {}
    
    # State accuracy (binary classification)
    state_pred = mx.argmax(state, axis=1)
    state_targ = mx.argmax(reachability_target, axis=1)
    state_acc = float(mx.mean((state_pred == state_targ).astype(mx.float32)))
    metrics['state_acc'] = state_acc
    
    # Predecessor accuracy (multi-class classification)
    pred_pred = mx.argmax(predesecor, axis=1)
    pred_acc = float(mx.mean((pred_pred == predesecor_target).astype(mx.float32)))
    metrics['pred_acc'] = pred_acc
    
    # Termination accuracy (binary classification)
    term_pred = mx.argmax(termination_prob, axis=0 if termination_prob.ndim == 1 else 1)
    term_targ = mx.argmax(termination_target, axis=0 if termination_target.ndim == 1 else 1)
    term_acc = float((term_pred == term_targ).astype(mx.float32))
    metrics['term_acc'] = term_acc
    
    # Distance accuracy (for parallel algorithms only)
    if distance is not None and distance_target is not None:
        # For distance, we calculate MAE and also a threshold-based accuracy
        distance_mae = float(mx.mean(mx.abs(distance.flatten() - distance_target.flatten())))
        # Consider distance "accurate" if within 10% of target (or 0.1 absolute for small values)
        distance_thresh = mx.maximum(mx.abs(distance_target.flatten()) * 0.1, 0.1)
        distance_within_thresh = mx.abs(distance.flatten() - distance_target.flatten()) <= distance_thresh
        distance_acc = float(mx.mean(distance_within_thresh.astype(mx.float32)))
        metrics['dist_mae'] = distance_mae
        metrics['dist_acc'] = distance_acc
    
    return metrics

class SimpleLogger:
    """Minimal logger with optional debug mode and accuracy tracking"""
    def __init__(self, debug=False):
        self.debug = debug
        # Track accuracy metrics for each epoch
        self.train_metrics = []
        self.val_metrics = []
        self.step_metrics = {'train': [], 'val': []}
    
    def start_epoch(self, epoch, num_epochs, total_graphs):
        """Start epoch (no progress bar)"""
        # Reset step metrics for new epoch
        self.step_metrics = {'train': [], 'val': []}
    
    def update_progress(self, train_loss=None, val_loss=None):
        """Update progress (no progress bar)"""
        pass
    
    def log_step_metrics(self, metrics, phase='train'):
        """Log metrics for a single step"""
        self.step_metrics[phase].append(metrics)
    
    def _average_metrics(self, metrics_list):
        """Average a list of metric dictionaries"""
        if not metrics_list:
            return {}
        
        # Get all keys from first metrics dict
        keys = metrics_list[0].keys()
        averaged = {}
        
        for key in keys:
            values = [m[key] for m in metrics_list if key in m]
            if values:
                averaged[key] = sum(values) / len(values)
        
        return averaged
    
    def log_epoch(self, epoch, train_loss, val_loss):
        """Log epoch results with accuracy metrics"""
        # Calculate average metrics for the epoch
        train_metrics = self._average_metrics(self.step_metrics['train'])
        val_metrics = self._average_metrics(self.step_metrics['val'])
        
        # Store metrics
        self.train_metrics.append(train_metrics)
        self.val_metrics.append(val_metrics)
        
        # Print epoch summary
        print(f"Epoch {epoch + 1}:")
        print(f"  Losses - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
        
        # Print accuracy metrics if available
        if train_metrics:
            print(f"  Train Acc - State: {train_metrics.get('state_acc', 0):.3f}, "
                  f"Pred: {train_metrics.get('pred_acc', 0):.3f}, "
                  f"Term: {train_metrics.get('term_acc', 0):.3f}", end="")
            if 'dist_acc' in train_metrics:
                print(f", Dist: {train_metrics.get('dist_acc', 0):.3f} (MAE: {train_metrics.get('dist_mae', 0):.3f})", end="")
            print()
        
        if val_metrics:
            print(f"  Val Acc   - State: {val_metrics.get('state_acc', 0):.3f}, "
                  f"Pred: {val_metrics.get('pred_acc', 0):.3f}, "
                  f"Term: {val_metrics.get('term_acc', 0):.3f}", end="")
            if 'dist_acc' in val_metrics:
                print(f", Dist: {val_metrics.get('dist_acc', 0):.3f} (MAE: {val_metrics.get('dist_mae', 0):.3f})", end="")
            print()
    
    def log_final(self, best_val_loss, best_epoch):
        """Log final training results"""
        print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}")
        
        # Print best validation accuracies if available
        if best_epoch < len(self.val_metrics) and self.val_metrics[best_epoch]:
            best_metrics = self.val_metrics[best_epoch]
            print(f"Best validation accuracies:")
            print(f"  State: {best_metrics.get('state_acc', 0):.3f}, "
                  f"Pred: {best_metrics.get('pred_acc', 0):.3f}, "
                  f"Term: {best_metrics.get('term_acc', 0):.3f}", end="")
            if 'dist_acc' in best_metrics:
                print(f", Dist: {best_metrics.get('dist_acc', 0):.3f}")
            else:
                print()
    
    def log_debug_info(self, state, predesecor, reachability_target, predesecor_target, 
                      termination_prob, termination_target, distance=None, distance_target=None, task_type=None):
        """Print debug information if debug mode is enabled"""
        if not self.debug:
            return
            
        task_name = "SEQ" if task_type == task.SEQUENTIAL_ALGORITHM else "PAR"
        print(f"  DEBUG ({task_name}):")
        print(f"    Pred: {np.array(mx.argmax(predesecor, axis=1))}")
        print(f"    Targ: {np.array(predesecor_target)}")
        print(f"    State: {np.array(mx.argmax(state, axis=1))}")
        print(f"    S_Targ: {np.array(mx.argmax(reachability_target, axis=1))}")
        
        if distance is not None and distance_target is not None:
            print(f"    Dist: {np.array(distance.flatten())}")
            print(f"    D_Targ: {np.array(distance_target.flatten())}")
        
        print(f"    Term: {float(mx.softmax(termination_prob, axis=0)[1]):.3f}")


def count_parameters(model):
    """Count model parameters"""
    def count_params(tree):
        if isinstance(tree, dict):
            return sum(count_params(v) for v in tree.values())
        elif hasattr(tree, 'size'):
            return tree.size
        else:
            return 0
    return count_params(model.parameters())


def print_model_info(model):
    """Print basic model information"""
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count:,}") 