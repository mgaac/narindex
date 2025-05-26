import mlx.core as mx
import mlx.nn as nn
import numpy as np
from enum import Enum

# Task enumeration (needed for logging)
class task(Enum):
    PARALLEL_ALGORITHM=0
    SEQUENTIAL_ALGORITHM=1

class SimpleLogger:
    """Minimal logger with optional debug mode"""
    def __init__(self, debug=False):
        self.debug = debug
    
    def start_epoch(self, epoch, num_epochs, total_graphs):
        """Start epoch (no progress bar)"""
        pass
    
    def update_progress(self, train_loss=None, val_loss=None):
        """Update progress (no progress bar)"""
        pass
    
    def log_epoch(self, epoch, train_loss, val_loss):
        """Log epoch results"""
        print(f"Epoch {epoch + 1}: Train={train_loss:.4f}, Val={val_loss:.4f}")
    
    def log_final(self, best_val_loss, best_epoch):
        """Log final training results"""
        print(f"Best: {best_val_loss:.4f} at epoch {best_epoch + 1}")
    
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