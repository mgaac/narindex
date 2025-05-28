"""
Neural Turing Machine (NTM) implementation in MLX.
Based on "Neural Turing Machines" (Graves et al., 2014)
arXiv:1410.5401v2

This implementation uses a branching architecture where controller and output 
branches are isolated and share a common preprocessing module.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Dict, Any


class NeuralTuringMachine(nn.Module):
    """
    Neural Turing Machine with external memory.
    
    The NTM consists of a controller network that can read from and write to
    an external memory matrix using learned attention mechanisms.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dim: int, 
        num_shared_layers: int, 
        num_controller_layers: int, 
        num_output_layers: int, 
        memory_size: Tuple[int, int]
    ):
        """
        Initialize Neural Turing Machine.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension  
            hidden_dim: Hidden layer dimension
            num_shared_layers: Number of shared preprocessing layers
            num_controller_layers: Number of controller-specific layers
            num_output_layers: Number of output-specific layers
            memory_size: Tuple of (memory_rows, memory_cols)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        
        # Calculate layer sizes
        memory_rows, memory_cols = memory_size
        shared_input_size = input_dim + memory_cols  # input + read vector
        
        # Controller output: addressing parameters
        # 3 * memory_cols for (add, erase, key) + 3 + 3 for (shift, beta, gamma)
        controller_output_size = memory_cols * 3 + 6
        
        # Shared preprocessing layers
        shared_layer_sizes = [shared_input_size] + [hidden_dim] * num_shared_layers
        self.shared_layers = [
            nn.Linear(in_size, out_size, bias=True) 
            for in_size, out_size in zip(shared_layer_sizes[:-1], shared_layer_sizes[1:])
        ]
        
        # Controller branch layers
        controller_layer_sizes = [hidden_dim] * num_controller_layers + [controller_output_size]
        self.controller_layers = [
            nn.Linear(in_size, out_size, bias=True)
            for in_size, out_size in zip(controller_layer_sizes[:-1], controller_layer_sizes[1:])
        ]
        
        # Output branch layers
        output_layer_sizes = [hidden_dim] * num_output_layers + [output_dim]
        self.output_layers = [
            nn.Linear(in_size, out_size, bias=True)
            for in_size, out_size in zip(output_layer_sizes[:-1], output_layer_sizes[1:])
        ]
    
    def __call__(
        self, 
        x: mx.array, 
        read_vector: mx.array, 
        write_weights: mx.array, 
        memory: mx.array
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array, Dict[str, mx.array]]:
        """
        Forward pass of Neural Turing Machine.
        
        Args:
            x: Input vector [input_dim]
            read_vector: Previous read vector [memory_cols]
            write_weights: Previous write weights [memory_rows]
            memory: Memory matrix [memory_rows, memory_cols]
            
        Returns:
            Tuple of (output, new_read_vector, new_write_weights, new_memory, activations)
        """
        activations = {}
        
        # Shared preprocessing
        shared_input = mx.concatenate([x, read_vector])
        shared_output = self._apply_layers(
            shared_input, self.shared_layers, "shared", activations, final_activation=True
        )
        
        # Controller branch - generates addressing parameters
        controller_output = self._apply_layers(
            shared_output, self.controller_layers, "controller", activations, final_activation=False
        )
        
        # Output branch - generates final output
        output = self._apply_layers(
            shared_output, self.output_layers, "output", activations, final_activation=False
        )
        
        # Parse controller outputs into addressing parameters
        memory_rows, memory_cols = self.memory_size
        addressing_params = self._parse_addressing_parameters(controller_output, memory_cols)
        
        # Store addressing parameters in activations
        for key, value in addressing_params.items():
            activations[key] = value
        
        # Perform memory operations
        new_write_weights = self._compute_addressing(
            addressing_params, write_weights, memory
        )
        new_read_vector = self._read_memory(new_write_weights, memory)
        new_memory = self._write_memory(
            new_write_weights, addressing_params['add_vector'], 
            addressing_params['erase_vector'], memory
        )
        
        activations['write_weights'] = new_write_weights
        activations['read_vector'] = new_read_vector
        activations['final_output'] = output
        
        return output, new_read_vector, new_write_weights, new_memory, activations
    
    def _apply_layers(
        self, 
        input_tensor: mx.array, 
        layers: list, 
        layer_name: str, 
        activations: Dict[str, mx.array],
        final_activation: bool = True
    ) -> mx.array:
        """Apply a sequence of layers with optional skip connections."""
        x = input_tensor
        
        for idx, layer in enumerate(layers):
            is_final = (idx == len(layers) - 1)
            
            if is_final and not final_activation:
                # Final layer without activation
                x = layer(x)
            else:
                # Apply layer with activation and optional skip connection
                pre_activation = layer(x)
                if idx > 0:  # Skip connection (except for first layer)
                    pre_activation += x
                x = nn.silu(pre_activation)
            
            activations[f"{layer_name}_layer_{idx}"] = x
        
        return x
    
    def _parse_addressing_parameters(
        self, 
        controller_output: mx.array, 
        memory_cols: int
    ) -> Dict[str, mx.array]:
        """Parse controller output into addressing parameters."""
        # Split controller output
        add_vector = controller_output[0:memory_cols]
        erase_vector = controller_output[memory_cols:2*memory_cols] 
        key_vector = controller_output[2*memory_cols:3*memory_cols]
        shift_weights = controller_output[3*memory_cols:3*memory_cols+3]
        beta = controller_output[3*memory_cols+3]
        gamma = controller_output[3*memory_cols+4]
        gate = controller_output[3*memory_cols+5]
        
        # Apply constraints to parameters
        return {
            'add_vector': mx.tanh(add_vector),
            'erase_vector': mx.tanh(erase_vector),
            'key_vector': mx.tanh(key_vector),
            'shift_weights': mx.softmax(shift_weights),
            'beta': mx.log(mx.exp(beta) + 1),  # Softplus
            'gamma': mx.log(mx.exp(gamma) + 1) + 1,  # Softplus + 1
            'gate': mx.sigmoid(gate)
        }
    
    def _compute_addressing(
        self, 
        params: Dict[str, mx.array], 
        prev_weights: mx.array, 
        memory: mx.array
    ) -> mx.array:
        """Compute addressing weights using content-based and location-based mechanisms."""
        # Content-based addressing
        memory_norms = mx.linalg.norm(memory, axis=1) + 1e-8
        key_norm = mx.linalg.norm(params['key_vector']) + 1e-8
        
        similarity = (params['key_vector'] @ memory.transpose()) / (key_norm * memory_norms)
        content_weights = mx.softmax(params['beta'] * similarity)
        
        # Interpolation between content and location
        gated_weights = (params['gate'] * content_weights + 
                        (1 - params['gate']) * prev_weights)
        
        # Convolutional shift
        shifted_weights = mx.convolve(gated_weights, params['shift_weights'], mode='same')
        
        # Sharpening
        sharpened_weights = mx.power(shifted_weights, params['gamma'])
        final_weights = sharpened_weights / sharpened_weights.sum()
        
        return mx.softmax(final_weights)
    
    def _read_memory(self, weights: mx.array, memory: mx.array) -> mx.array:
        """Read from memory using attention weights."""
        return weights @ memory
    
    def _write_memory(
        self, 
        weights: mx.array, 
        add_vector: mx.array, 
        erase_vector: mx.array, 
        memory: mx.array
    ) -> mx.array:
        """Write to memory using erase and add operations."""
        # Erase operation
        erase_matrix = mx.outer(weights, erase_vector)
        erased_memory = memory * (1 - erase_matrix)
        
        # Add operation
        add_matrix = mx.outer(weights, add_vector)
        new_memory = erased_memory + add_matrix
        
        # Apply sigmoid to keep values bounded
        return mx.sigmoid(new_memory)


# Alias for backward compatibility
controller = NeuralTuringMachine
