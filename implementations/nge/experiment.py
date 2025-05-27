#!/usr/bin/env python3
"""
Neural Execution of Graph Algorithms (NGE) Experiment Runner
Neural Graph Execution for Algorithmic Reasoning

Usage:
    python experiment.py --epochs 50 --task sequential
    python experiment.py --epochs 100 --task both --aggregation sum
"""

import argparse
import os
import pickle
from pathlib import Path
import mlx.core as mx
from tqdm import tqdm
import numpy as np

from model import task, nge, aggregation_fn
from train import create_trainer
from nge_utils import SimpleLogger, calculate_accuracy_metrics


def print_experiment_header(**config):
    """Print NGE experiment header with configuration."""
    print(f"\n{'='*75}")
    print(f"Neural Execution of Graph Algorithms (NGE) Experiment")
    print(f"Learning to Execute: Prim's MST & Breadth-First Search")
    print(f"{'='*75}")
    for key, value in config.items():
        print(f"{key}: {value}")
    print(f"{'='*75}")


def debug_algorithm_execution(trainer, dataset, task_types):
    """Debug algorithm execution by showing step-by-step execution on a validation graph."""
    print(f"\n{'='*75}")
    print(f"ALGORITHM EXECUTION DEBUG MODE")
    print(f"{'='*75}")
    
    # Take first graph from validation set for debugging
    debug_graph = dataset[0]
    
    for task_type in task_types:
        task_name = "sequential" if task_type == task.SEQUENTIAL_ALGORITHM else "parallel"
        algorithm_name = "Prim's MST" if task_type == task.SEQUENTIAL_ALGORITHM else "Breadth-First Search"
        task_type_int = task_type.value if hasattr(task_type, 'value') else task_type
        
        print(f"\n{'-'*50}")
        print(f"DEBUGGING: {algorithm_name} ({task_name.upper()})")
        print(f"{'-'*50}")
        
        if task_type_int == 0:  # PARALLEL_ALGORITHM
            execution_history = debug_graph['targets']['parallel']
            state_key = 'bfs_state'
            pred_key = 'bf_predecessor'
            term_key = 'bf_termination'
            distance_key = 'bf_distance'
        else:  # SEQUENTIAL_ALGORITHM
            execution_history = debug_graph['targets']['sequential']
            state_key = 'prim_state'
            pred_key = 'prim_predecessor'
            term_key = 'prim_termination'
            distance_key = None
        
        connection_matrix = debug_graph['connection_matrix']
        num_nodes = connection_matrix.shape[0]
        num_steps = len(execution_history[state_key]) - 1
        
        print(f"Graph Info: {num_nodes} nodes, {num_steps} algorithm steps")
        
        if num_steps == 0:
            print("No execution steps to debug.")
            continue
        
        # Step through algorithm execution
        residual_features = mx.zeros([num_nodes])
        
        for step_idx in range(num_steps):  # Show all steps
            print(f"\nStep {step_idx + 1}/{num_steps}:")
            print(f"{'~'*30}")
            
            # Current state
            current_state = execution_history[state_key][step_idx]
            target_state = execution_history[state_key][step_idx + 1]
            target_pred = execution_history[pred_key][step_idx + 1]
            target_term = execution_history[term_key][step_idx + 1]
            
            # Safe conversion functions
            def safe_argmax(arr):
                try:
                    np_arr = np.array(arr)
                    if np_arr.ndim == 1:
                        return np_arr
                    elif np_arr.ndim == 2:
                        return np.argmax(np_arr, axis=1)
                    else:
                        return np_arr.flatten()
                except:
                    return str(arr)[:50] + "..."
            
            def safe_scalar(val):
                """Safely convert to scalar."""
                try:
                    if hasattr(val, 'item'):
                        # Try to convert MLX array to scalar
                        val_np = np.array(val)
                        if val_np.size == 1:
                            return float(val_np.item())
                        else:
                            return float(val_np.flatten()[0])
                    elif isinstance(val, (list, np.ndarray)):
                        val_arr = np.array(val)
                        if val_arr.size == 1:
                            return float(val_arr.item())
                        else:
                            return float(val_arr.flatten()[0])
                    else:
                        return float(val)
                except Exception:
                    # Fallback: just return 0.0 if conversion fails
                    return 0.0
            
            current_state_disp = safe_argmax(current_state)
            target_state_disp = safe_argmax(target_state)
            target_pred_disp = safe_argmax(target_pred)
            target_term_val = safe_scalar(target_term[1])
            
            print(f"Current State: {current_state_disp}")
            print(f"Target State:  {target_state_disp}")
            print(f"Target Pred:   {target_pred_disp}")
            print(f"Target Term:   {target_term_val:.3f}")
            
            # Model prediction
            try:
                current_features = mx.argmax(current_state, axis=1) if len(current_state.shape) > 1 else current_state
                # Ensure residual_features matches the length of current_features
                if len(current_features) != len(residual_features):
                    residual_features = mx.zeros(len(current_features))
                input_features = mx.stack([current_features, residual_features], axis=1)
                input_data = (input_features, connection_matrix)
                
                if distance_key:
                    target_distance = execution_history[distance_key][step_idx + 1]
                    graph_targets = (target_state, target_distance, target_pred)
                    target_distance_disp = safe_argmax(target_distance)
                    print(f"Target Dist:   {target_distance_disp}")
                else:
                    graph_targets = (target_state, target_pred)
                
                # Get model predictions
                loss, losses, output, termination_prob = trainer.eval_step(
                    input_data, graph_targets, target_term, task_type_int, logger=None
                )
                
                # Show predictions
                if task_type_int == 0:  # PARALLEL_ALGORITHM
                    pred_state, pred_distance, pred_pred = output
                    pred_state_disp = safe_argmax(pred_state)
                    pred_distance_disp = safe_argmax(pred_distance)
                    pred_pred_disp = safe_argmax(pred_pred)
                    print(f"Pred State:    {pred_state_disp}")
                    print(f"Pred Dist:     {pred_distance_disp}")
                    print(f"Pred Pred:     {pred_pred_disp}")
                else:  # SEQUENTIAL_ALGORITHM
                    pred_state, pred_pred = output
                    pred_state_disp = safe_argmax(pred_state)
                    pred_pred_disp = safe_argmax(pred_pred)
                    print(f"Pred State:    {pred_state_disp}")
                    print(f"Pred Pred:     {pred_pred_disp}")
                
                print(f"Pred Term:     {safe_scalar(mx.softmax(termination_prob)[1]):.3f}")
                print(f"Step Loss:     {safe_scalar(loss):.4f}")
                
                # Calculate step accuracy
                if task_type_int == 0:
                    metrics = calculate_accuracy_metrics(
                        pred_state, pred_pred, target_state, target_pred,
                        termination_prob, target_term, pred_distance, target_distance
                    )
                else:
                    metrics = calculate_accuracy_metrics(
                        pred_state, pred_pred, target_state, target_pred,
                        termination_prob, target_term
                    )
                
                print(f"Step Accuracy: state={metrics['state_acc']:.3f}, pred={metrics['pred_acc']:.3f}, term={metrics['term_acc']:.3f}")
                if 'dist_acc' in metrics:
                    print(f"               dist={metrics['dist_acc']:.3f}")
                    
            except Exception as e:
                print(f"Error in model prediction: {e}")
                print("Skipping this step...")
                continue


def run_experiment(args):
    """Run NGE training experiment with algorithm execution analysis."""
    
    # Map task and aggregation
    task_map = {
        'sequential': task.SEQUENTIAL_ALGORITHM,
        'parallel': task.PARALLEL_ALGORITHM,
        'both': 'both'
    }
    
    agg_map = {
        'sum': aggregation_fn.SUM,
        'avg': aggregation_fn.AVG,
        'max': aggregation_fn.MAX,
        'min': aggregation_fn.MIN
    }
    
    task_type = task_map[args.task]
    aggregation_enum = agg_map[args.aggregation]
    
    # Setup task types
    if task_type == 'both':
        task_types = [task.SEQUENTIAL_ALGORITHM, task.PARALLEL_ALGORITHM]
        task_names = ['sequential (Prim\'s MST)', 'parallel (BFS)']
        algorithms = ['Prim\'s Minimum Spanning Tree', 'Breadth-First Search']
    else:
        task_types = [task_type]
        if task_type == task.SEQUENTIAL_ALGORITHM:
            task_names = ['sequential (Prim\'s MST)']
            algorithms = ['Prim\'s Minimum Spanning Tree']
        else:
            task_names = ['parallel (BFS)']
            algorithms = ['Breadth-First Search']
    
    # Configuration
    model_config = {
        'embedding_dim': args.embedding_dim,
        'dropout_prob': args.dropout,
        'skip_connections': args.skip_connections,
        'aggregation_fn': aggregation_enum,
        'num_mp_layers': args.mp_layers
    }
    
    experiment_config = {
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'early_stopping_patience': args.patience,
        'dataset': 'NEGA Custom (Graph Algorithm Execution)',
        'algorithms': ', '.join(algorithms),
        'architecture': f"{args.mp_layers} MP layers, {args.embedding_dim}d embedding",
        'aggregation': f"{args.aggregation.upper()} aggregation function",
        'regularization': f"dropout={args.dropout}, skip_conn={args.skip_connections}"
    }
    
    print_experiment_header(**experiment_config)
    
    # Create trainer
    trainer = create_trainer(model_config, learning_rate=args.learning_rate)
    
    print(f"Model initialized for graph algorithm execution")
    
    # Load datasets - including test set
    datasets = {}
    data_files = ['train_graphs.pkl', 'val_graphs.pkl', 'test_graphs.pkl']
    for data_file in data_files:
        dataset_path = f'../../utils/datasets/nega_custom/data/{data_file}'
        if os.path.exists(dataset_path):
            with open(dataset_path, 'rb') as f:
                key = data_file.replace('_graphs.pkl', '')
                datasets[key] = pickle.load(f)
        else:
            if data_file != 'test_graphs.pkl':  # Test set might not exist
                print(f"Warning: {dataset_path} not found")
    
    if 'train' not in datasets or 'val' not in datasets:
        print("Error: Required datasets not found. Please ensure NEGA custom data is available.")
        return None
    
    print(f"\nDataset Statistics:")
    print(f"  Training graphs: {len(datasets['train']):,}")
    print(f"  Validation graphs: {len(datasets['val']):,}")
    if 'test' in datasets:
        print(f"  Test graphs: {len(datasets['test']):,}")
    print(f"  Algorithms to learn: {', '.join(algorithms)}")
    
    # Training tracking with deltas
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    train_history = []
    val_history = []
    
    # Previous values for delta calculation
    prev_train_loss = None
    prev_val_loss = None
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Tasks: {', '.join(task_names)}")
    
    # Create custom progress tracking
    epoch_pbar = tqdm(range(args.epochs), desc="Training NGE", unit="epoch",
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for epoch in range(args.epochs):
        # Training phase
        train_losses = []
        
        for task_idx, current_task in enumerate(task_types):
            # Train on this task
            train_loss = trainer.train_model(datasets['train'], current_task, logger=None, phase="train")
            train_losses.append(train_loss)
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        
        # Validation phase
        val_losses = []
        
        for task_idx, current_task in enumerate(task_types):
            # Validate on this task
            val_loss = trainer.train_model(datasets['val'], current_task, logger=None, phase="val")
            val_losses.append(val_loss)
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        # Calculate deltas
        train_delta = 0.0 if prev_train_loss is None else avg_train_loss - prev_train_loss
        val_delta = 0.0 if prev_val_loss is None else avg_val_loss - prev_val_loss
        
        # Store history
        train_history.append(avg_train_loss)
        val_history.append(avg_val_loss)
        
        # Update progress with deltas
        postfix_dict = {
            'train_loss': f"{avg_train_loss:.4f}",
            'val_loss': f"{avg_val_loss:.4f}",
            'best_epoch': best_epoch + 1,
            'patience': f"{patience_counter}/{args.patience}"
        }
        
        if prev_train_loss is not None:
            postfix_dict['Δtrain'] = f"{train_delta:+.4f}"
            postfix_dict['Δval'] = f"{val_delta:+.4f}"
        
        epoch_pbar.set_postfix(postfix_dict)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            epoch_pbar.set_description("Early stopping")
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break
        
        # Update previous values
        prev_train_loss = avg_train_loss
        prev_val_loss = avg_val_loss
        
        # Print per-epoch summary
        print(f"\nEpoch {epoch + 1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Best: {best_val_loss:.4f} (epoch {best_epoch + 1})")
        
        epoch_pbar.update(1)
    
    epoch_pbar.close()
    
    # Final evaluation on all datasets
    print(f"\n{'='*75}")
    print(f"NGE Training Completed!")
    print(f"{'='*75}")
    print(f"Training Results:")
    print(f"  Best Validation Loss: {best_val_loss:.4f} (epoch {best_epoch + 1})")
    print(f"  Total Epochs: {epoch + 1}")
    print(f"  Final Training Loss: {train_history[-1]:.4f}")
    print(f"  Final Validation Loss: {val_history[-1]:.4f}")
    
    # Test on all available datasets
    print(f"\nFinal Evaluation on All Datasets:")
    for dataset_name, dataset in datasets.items():
        if dataset_name == 'train':
            continue  # Skip training set for final eval
            
        print(f"\n{dataset_name.capitalize()} Set Evaluation:")
        for task_idx, current_task in enumerate(task_types):
            task_name = "Sequential (Prim's MST)" if current_task == task.SEQUENTIAL_ALGORITHM else "Parallel (BFS)"
            test_loss = trainer.train_model(dataset, current_task, logger=None, phase="val")
            print(f"  {task_name}: Loss = {test_loss:.4f}")
    
    # Algorithm execution debug mode
    if args.analyze_execution:
        debug_algorithm_execution(trainer, datasets['val'], task_types)
    
    return {
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'final_train_loss': train_history[-1],
        'final_val_loss': val_history[-1],
        'train_history': train_history,
        'val_history': val_history,
        'task_types': [t.name if hasattr(t, 'name') else str(t) for t in task_types],
        'model_config': model_config
    }


def main():
    parser = argparse.ArgumentParser(
        description="Neural Execution of Graph Algorithms (NGE) Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single algorithm training
  python experiment.py --task sequential --epochs 100
  python experiment.py --task parallel --epochs 100
  
  # Multi-task learning
  python experiment.py --task both --epochs 150
  
  # Architecture exploration
  python experiment.py --mp-layers 3 --embedding-dim 64
  
  # Debug algorithm execution step-by-step
  python experiment.py --task both --epochs 50 --analyze-execution
  
NGE Architecture Details:
  - Neural execution of classical graph algorithms
  - Multi-task learning: Prim's MST + Breadth-First Search
  - Message passing with multiple aggregation functions
  - Step-by-step algorithm execution learning
  
Algorithms:
  - Sequential: Prim's Minimum Spanning Tree algorithm
  - Parallel: Breadth-First Search algorithm
  - Multi-task: Learn both algorithms simultaneously
  
Debug Mode (--analyze-execution):
  - Shows step-by-step algorithm execution on validation graph
  - Displays intermediate states, predictions, and targets
  - Useful for understanding model behavior and debugging
        """
    )
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    
    # NGE-specific parameters
    parser.add_argument('--task', choices=['sequential', 'parallel', 'both'], default='sequential',
                        help='Algorithm task type (default: sequential)')
    parser.add_argument('--aggregation', choices=['sum', 'avg', 'max', 'min'], default='max',
                        help='Message passing aggregation function (default: max)')
    parser.add_argument('--mp-layers', type=int, default=2,
                        help='Number of message passing layers (default: 2)')
    parser.add_argument('--embedding-dim', type=int, default=32,
                        help='Node embedding dimension (default: 32)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability (default: 0.5)')
    parser.add_argument('--skip-connections', action='store_true', default=True,
                        help='Use skip connections (default: True)')
    parser.add_argument('--no-skip-connections', action='store_false', dest='skip_connections',
                        help='Disable skip connections')
    
    # Analysis options
    parser.add_argument('--analyze-execution', action='store_true',
                        help='Debug mode: show step-by-step algorithm execution on validation graph')
    
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