import argparse
import sys
import os
import pickle

# Add the utils directory to the path for dataset loading
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from model import task, aggregation_fn
from train import create_trainer
from nge_utils import SimpleLogger, print_model_info


def load_graphs(filename):
    """Load graphs from the dataset directory"""
    filepath = os.path.join('..', '..', 'utils', 'datasets', 'nega_custom', 'data', filename)
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def load_all_datasets():
    """Load all available datasets"""
    datasets = {
        'train': load_graphs('train_graphs.pkl'),
        'val': load_graphs('val_graphs.pkl'),
        'test_20': load_graphs('test_graphs_20.pkl'),
        'test_50': load_graphs('test_graphs_50.pkl'),
        'test_100': load_graphs('test_graphs_100.pkl')
    }
    return datasets


def run_experiment(task_types, epochs=10, debug=False, test_sizes='all'):
    """Run experiment with automatic testing after training"""
    
    # Simple configuration
    model_config = {
        'embedding_dim': 100,
        'dim_proj': 10,
        'dropout_prob': 0.5,
        'skip_connections': True,
        'aggregation_fn': aggregation_fn.MAX,
        'num_mp_layers': 1
    }
    
    # Create trainer and logger
    trainer = create_trainer(model_config, learning_rate=0.001)
    logger = SimpleLogger(debug=debug)
    
    # Load datasets
    datasets = load_all_datasets()
    train_graphs = datasets['train']
    val_graphs = datasets['val']
    
    print(f"Loaded {len(train_graphs)} train, {len(val_graphs)} val graphs")
    print_model_info(trainer.model)
    
    # Determine task types
    if isinstance(task_types, str):
        if task_types == 'both':
            task_list = [task.SEQUENTIAL_ALGORITHM, task.PARALLEL_ALGORITHM]
            print("Training on both sequential and parallel tasks")
        elif task_types == 'sequential':
            task_list = [task.SEQUENTIAL_ALGORITHM]
            print("Training on sequential task")
        elif task_types == 'parallel':
            task_list = [task.PARALLEL_ALGORITHM]
            print("Training on parallel task")
    else:
        task_list = task_types
    
    # Run training
    results = trainer.train_harness(
        train_graphs,
        val_graphs, 
        logger,
        task_types=task_list,
        num_epochs=epochs,
        early_stopping_patience=3
    )
    
    # Always evaluate on test sets
    test_datasets = {}
    if test_sizes == 'all':
        test_datasets = {
            'test_20': datasets['test_20'],
            'test_50': datasets['test_50'], 
            'test_100': datasets['test_100']
        }
    else:
        # Parse specific test sizes
        for size in test_sizes.split(','):
            size = size.strip()
            if size in ['20', '50', '100']:
                test_datasets[f'test_{size}'] = datasets[f'test_{size}']
    
    if test_datasets:
        test_results = trainer.evaluate_on_test_sets(test_datasets, task_list)
        results['test_results'] = test_results
    
    return results


def main():
    parser = argparse.ArgumentParser(description="NGE Experiment Runner")
    parser.add_argument('--task', required=True, choices=['sequential', 'parallel', 'both'],
                        help='Task type: sequential, parallel, or both')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--test', default='all', 
                        help='Test datasets: all, 20, 50, 100, or comma-separated (e.g., 20,50)')
    
    args = parser.parse_args()
    
    print(f"Running {args.task} task(s) for {args.epochs} epochs")
    
    results = run_experiment(args.task, args.epochs, args.debug, args.test)
    
    print("\nTraining completed!")
    
    # Always print test results
    print("\nTEST RESULTS:")
    for dataset_name, dataset_results in results['test_results'].items():
        print(f"{dataset_name}: ", end="")
        losses = [f"{task}={loss:.3f}" for task, loss in dataset_results.items()]
        print(", ".join(losses))
        

if __name__ == "__main__":
    exit(main()) 