"""
Neural Turing Machine (NTM) Model

Paper: "Neural Turing Machines" (Graves et al., 2014)
       arXiv:1410.5401v2

Implements a neural network with external memory using content-based
and location-based addressing mechanisms.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Dict


class NeuralTuringMachine(nn.Module):
    """
    Neural Turing Machine with external memory.
    
    Uses a branching architecture where controller and output branches
    share a common preprocessing module.
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
        
        memory_rows, memory_cols = memory_size
        shared_input_size = input_dim + memory_cols
        
        # Controller output: addressing parameters
        controller_output_size = memory_cols * 3 + 6
        
        # Shared preprocessing layers
        shared_sizes = [shared_input_size] + [hidden_dim] * num_shared_layers
        self.shared_layers = [
            nn.Linear(in_size, out_size, bias=True) 
            for in_size, out_size in zip(shared_sizes[:-1], shared_sizes[1:])
        ]
        
        # Controller branch layers
        controller_sizes = [hidden_dim] * num_controller_layers + [controller_output_size]
        self.controller_layers = [
            nn.Linear(in_size, out_size, bias=True)
            for in_size, out_size in zip(controller_sizes[:-1], controller_sizes[1:])
        ]
        
        # Output branch layers
        output_sizes = [hidden_dim] * num_output_layers + [output_dim]
        self.output_layers = [
            nn.Linear(in_size, out_size, bias=True)
            for in_size, out_size in zip(output_sizes[:-1], output_sizes[1:])
        ]
    
    def __call__(
        self, 
        x: mx.array, 
        read_vector: mx.array, 
        write_weights: mx.array, 
        memory: mx.array
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array, Dict[str, mx.array]]:
        """
        Forward pass.
        
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
        
        # Controller branch
        controller_output = self._apply_layers(
            shared_output, self.controller_layers, "controller", activations, final_activation=False
        )
        
        # Output branch
        output = self._apply_layers(
            shared_output, self.output_layers, "output", activations, final_activation=False
        )
        
        # Parse addressing parameters
        memory_rows, memory_cols = self.memory_size
        addressing_params = self._parse_addressing_parameters(controller_output, memory_cols)
        
        for key, value in addressing_params.items():
            activations[key] = value
        
        # Memory operations
        new_write_weights = self._compute_addressing(addressing_params, write_weights, memory)
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
        x: mx.array, 
        layers: list, 
        name: str, 
        activations: Dict[str, mx.array],
        final_activation: bool = True
    ) -> mx.array:
        """Apply a sequence of layers with optional skip connections."""
        for idx, layer in enumerate(layers):
            is_final = (idx == len(layers) - 1)
            
            if is_final and not final_activation:
                x = layer(x)
            else:
                pre_activation = layer(x)
                if idx > 0:
                    pre_activation = pre_activation + x
                x = nn.silu(pre_activation)
            
            activations[f"{name}_layer_{idx}"] = x
        
        return x
    
    def _parse_addressing_parameters(
        self, 
        controller_output: mx.array, 
        memory_cols: int
    ) -> Dict[str, mx.array]:
        """Parse controller output into addressing parameters."""
        add_vector = controller_output[0:memory_cols]
        erase_vector = controller_output[memory_cols:2*memory_cols] 
        key_vector = controller_output[2*memory_cols:3*memory_cols]
        shift_weights = controller_output[3*memory_cols:3*memory_cols+3]
        beta = controller_output[3*memory_cols+3]
        gamma = controller_output[3*memory_cols+4]
        gate = controller_output[3*memory_cols+5]
        
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
        """Compute addressing weights using content and location mechanisms."""
        # Content-based addressing
        memory_norms = mx.linalg.norm(memory, axis=1) + 1e-8
        key_norm = mx.linalg.norm(params['key_vector']) + 1e-8
        
        similarity = (params['key_vector'] @ memory.transpose()) / (key_norm * memory_norms)
        content_weights = mx.softmax(params['beta'] * similarity)
        
        # Interpolation
        gated_weights = (params['gate'] * content_weights + 
                        (1 - params['gate']) * prev_weights)
        
        # Convolutional shift
        shifted_weights = mx.convolve(gated_weights, params['shift_weights'], mode='same')
        
        # Sharpening
        sharpened = mx.power(shifted_weights, params['gamma'])
        final_weights = sharpened / sharpened.sum()
        
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
        erase_matrix = mx.outer(weights, erase_vector)
        erased_memory = memory * (1 - erase_matrix)
        
        add_matrix = mx.outer(weights, add_vector)
        new_memory = erased_memory + add_matrix
        
        return mx.sigmoid(new_memory)
