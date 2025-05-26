import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as optim
from functools import partial

from model import task, nge, PARALLEL_ALGORITHM, SEQUENTIAL_ALGORITHM


class NGETrainer:    
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

        self.state = [model.state, optimizer.state, mx.random.state]
        
        # Create gradient functions
        self.sequential_loss_and_grad_fn = nn.value_and_grad(model, self._sequential_loss_fn)
        self.parallel_loss_and_grad_fn = nn.value_and_grad(model, self._parallel_loss_fn)
        
        # Compile separate functions for each task type to avoid conditional logic in compiled functions
        self.compiled_sequential_train_step = mx.compile(self._sequential_train_step_impl, inputs=self.state, outputs=self.state)
        self.compiled_parallel_train_step = mx.compile(self._parallel_train_step_impl, inputs=self.state, outputs=self.state)
        self.compiled_sequential_eval_step = mx.compile(self._sequential_eval_step_impl, inputs=self.state, outputs=self.state)
        self.compiled_parallel_eval_step = mx.compile(self._parallel_eval_step_impl, inputs=self.state, outputs=self.state)
    
    def _sequential_loss_fn(self, model, input_data, graph_targets, termination_target):
        """Loss function for sequential algorithms (Prim's MST)"""
        output, termination_prob = model(input_data, SEQUENTIAL_ALGORITHM)
        state, predesecor = output
        reachability_target, predesecor_target = graph_targets

        state_loss = nn.losses.binary_cross_entropy(state, reachability_target, reduction='mean')
        pred_loss = nn.losses.cross_entropy(predesecor, predesecor_target, reduction='mean')
        termination_loss = nn.losses.binary_cross_entropy(termination_prob, termination_target, reduction='mean')
        total_loss = state_loss + pred_loss + termination_loss

        return total_loss, (state_loss, pred_loss, termination_loss), output, termination_prob

    def _parallel_loss_fn(self, model, input_data, graph_targets, termination_target):
        """Loss function for parallel algorithms (BFS)"""
        output, termination_prob = model(input_data, PARALLEL_ALGORITHM)
        state, distance, predesecor = output
        reachability_target, distance_target, predesecor_target = graph_targets

        state_loss = nn.losses.binary_cross_entropy(state, reachability_target, reduction='mean')
        distance_loss = nn.losses.mse_loss(distance.squeeze(), distance_target, reduction='mean')
        pred_loss = nn.losses.cross_entropy(predesecor, predesecor_target, reduction='mean')
        termination_loss = nn.losses.binary_cross_entropy(termination_prob, termination_target, reduction='mean')
        total_loss = state_loss + distance_loss + pred_loss + termination_loss

        return total_loss, (state_loss, distance_loss, pred_loss, termination_loss), output, termination_prob

    def _sequential_train_step_impl(self, input_data, graph_targets, termination_target, task_type):
        (loss, losses, output, termination_prob), grads = self.sequential_loss_and_grad_fn(self.model, input_data, graph_targets, termination_target)
        
        self.optimizer.update(self.model, grads)
        return loss, losses, output, termination_prob

    def _parallel_train_step_impl(self, input_data, graph_targets, termination_target, task_type):
        (loss, losses, output, termination_prob), grads = self.parallel_loss_and_grad_fn(self.model, input_data, graph_targets, termination_target)
        
        self.optimizer.update(self.model, grads)
        return loss, losses, output, termination_prob

    def _sequential_eval_step_impl(self, input_data, graph_targets, termination_target, task_type):
        return self._sequential_loss_fn(self.model, input_data, graph_targets, termination_target)

    def _parallel_eval_step_impl(self, input_data, graph_targets, termination_target, task_type):
        return self._parallel_loss_fn(self.model, input_data, graph_targets, termination_target)

    def train_step(self, input_data, graph_targets, termination_target, task_type, logger=None):
        if task_type == SEQUENTIAL_ALGORITHM:
            result = self.compiled_sequential_train_step(input_data, graph_targets, termination_target, task_type)
        elif task_type == PARALLEL_ALGORITHM:
            result = self.compiled_parallel_train_step(input_data, graph_targets, termination_target, task_type)
        
        # Evaluate outside of compiled function
        mx.eval(result, self.model.parameters())
        
        # Handle logging outside of compiled function
        if logger:
            loss, losses, output, termination_prob = result
            if task_type == PARALLEL_ALGORITHM:
                state, distance, predesecor = output
                reachability_target, distance_target, predesecor_target = graph_targets
                logger.log_debug_info(state, predesecor, reachability_target, predesecor_target, 
                                    termination_prob, termination_target, distance, distance_target, task.PARALLEL_ALGORITHM)
            elif task_type == SEQUENTIAL_ALGORITHM:
                state, predesecor = output
                reachability_target, predesecor_target = graph_targets
                logger.log_debug_info(state, predesecor, reachability_target, predesecor_target, 
                                    termination_prob, termination_target, task_type=task.SEQUENTIAL_ALGORITHM)
        
        return result

    def eval_step(self, input_data, graph_targets, termination_target, task_type, logger=None):
        if task_type == SEQUENTIAL_ALGORITHM:
            result = self.compiled_sequential_eval_step(input_data, graph_targets, termination_target, task_type)
        elif task_type == PARALLEL_ALGORITHM:
            result = self.compiled_parallel_eval_step(input_data, graph_targets, termination_target, task_type)
        
        # Handle logging outside of compiled function
        if logger:
            loss, losses, output, termination_prob = result
            if task_type == PARALLEL_ALGORITHM:
                state, distance, predesecor = output
                reachability_target, distance_target, predesecor_target = graph_targets
                logger.log_debug_info(state, predesecor, reachability_target, predesecor_target, 
                                    termination_prob, termination_target, distance, distance_target, task.PARALLEL_ALGORITHM)
            elif task_type == SEQUENTIAL_ALGORITHM:
                state, predesecor = output
                reachability_target, predesecor_target = graph_targets
                logger.log_debug_info(state, predesecor, reachability_target, predesecor_target, 
                                    termination_prob, termination_target, task_type=task.SEQUENTIAL_ALGORITHM)
        
        return result

    def train_model(self, dataset, task_type, logger=None, phase="train"):
        """Training function for both parallel and sequential algorithms"""
        is_train = (phase == "train")
        total_loss = 0.0
        valid_graphs = 0
        
        # Convert enum to int for compiled functions
        task_type_int = task_type.value if hasattr(task_type, 'value') else task_type
        
        for graph_idx, graph_data in enumerate(dataset):
            if task_type_int == PARALLEL_ALGORITHM:
                execution_history = graph_data['targets']['parallel']
                state_key = 'bfs_state'
                pred_key = 'bf_predecessor'
                term_key = 'bf_termination'
                distance_key = 'bf_distance'
            else:
                execution_history = graph_data['targets']['sequential']
                state_key = 'prim_state'
                pred_key = 'prim_predecessor'
                term_key = 'prim_termination'
                distance_key = None
            
            connection_matrix = graph_data['connection_matrix']
            residual_features = mx.zeros([len(execution_history[state_key][0])])
            num_steps = len(execution_history[state_key]) - 1

            if num_steps == 0:
                continue
            
            valid_graphs += 1
            graph_total_loss = 0.0
            
            for i in range(num_steps):
                # Prepare data
                state_target = execution_history[state_key][i + 1]
                pred_target = execution_history[pred_key][i + 1]
                termination_target = execution_history[term_key][i + 1]
                
                current_features = mx.argmax(execution_history[state_key][i], axis=1)
                input_features = mx.stack([current_features, residual_features], axis=1)
                input_data = (input_features, connection_matrix)
                
                if distance_key:
                    distance_target = execution_history[distance_key][i + 1]
                    graph_targets = (state_target, distance_target, pred_target)
                else:
                    graph_targets = (state_target, pred_target)
                
                # Training or evaluation step - use integer task type
                if is_train:
                    loss, losses, output, termination_prob = \
                        self.train_step(input_data, graph_targets, termination_target, task_type_int, logger)
                else:
                    loss, losses, output, termination_prob = \
                        self.eval_step(input_data, graph_targets, termination_target, task_type_int, logger)

                # Update residual features
                if task_type_int == PARALLEL_ALGORITHM:
                    state, distance, _ = output
                else:
                    state, _ = output
                residual_features = mx.argmax(state, axis=1)
                graph_total_loss += float(loss)
            
            # Update progress bar
            avg_graph_loss = graph_total_loss / num_steps
            if logger:
                if is_train:
                    logger.update_progress(train_loss=avg_graph_loss)
                else:
                    logger.update_progress(val_loss=avg_graph_loss)
            
            total_loss += avg_graph_loss
        
        return total_loss / valid_graphs if valid_graphs > 0 else 0.0

    def train_harness(self, train_dataset, val_dataset, logger=None, task_types=None, 
                     num_epochs=10, early_stopping_patience=3):
        """Main training harness supporting single or multi-task training"""
        if task_types is None:
            task_types = [task.SEQUENTIAL_ALGORITHM]
        elif not isinstance(task_types, list):
            task_types = [task_types]
            
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Start epoch progress bar
            if logger:
                total_graphs = len(train_dataset) + len(val_dataset)
                logger.start_epoch(epoch, num_epochs, total_graphs)
            
            # Training phase - alternate between tasks if multi-task
            train_losses = []
            for task_type in task_types:
                train_loss = self.train_model(train_dataset, task_type, logger, phase="train")
                train_losses.append(train_loss)
            avg_train_loss = sum(train_losses) / len(train_losses)
            
            # Validation phase - evaluate on all tasks
            val_losses = []
            for task_type in task_types:
                val_loss = self.train_model(val_dataset, task_type, logger, phase="val")
                val_losses.append(val_loss)
            avg_val_loss = sum(val_losses) / len(val_losses)
            
            # Log epoch
            if logger:
                logger.log_epoch(epoch, avg_train_loss, avg_val_loss)
            
            # Check for improvement
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Log final results
        if logger:
            logger.log_final(best_val_loss, best_epoch)
        
        return {
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'task_types': task_types,
            'train_losses': train_losses,
            'val_losses': val_losses
        }

    def evaluate_on_test_sets(self, test_datasets, task_types=None, logger=None):
        """Evaluate model on multiple test datasets"""
        if task_types is None:
            task_types = [task.SEQUENTIAL_ALGORITHM, task.PARALLEL_ALGORITHM]
        elif not isinstance(task_types, list):
            task_types = [task_types]
            
        results = {}
        
        for dataset_name, dataset in test_datasets.items():
            print(f"\nEvaluating on {dataset_name} ({len(dataset)} graphs)...")
            dataset_results = {}
            
            for task_type in task_types:
                task_name = "sequential" if task_type == task.SEQUENTIAL_ALGORITHM else "parallel"
                test_loss = self.train_model(dataset, task_type, logger, phase="test")
                dataset_results[task_name] = test_loss
                print(f"  {task_name}: {test_loss:.4f}")
            
            results[dataset_name] = dataset_results
        
        return results


def create_trainer(model_config, learning_rate=0.001):
    """Factory function to create a trainer with model and optimizer"""
    model = nge(**model_config)
    optimizer = optim.Adam(learning_rate=learning_rate)
    return NGETrainer(model, optimizer)