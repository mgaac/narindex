"""
Message Passing Neural Network (MPNN) implementation in MLX.
Based on "Neural Message Passing for Quantum Chemistry" (Gilmer et al., 2017)
arXiv:1704.01212v2
"""

import mlx.core as mx
import mlx.nn as nn
from enum import Enum
from typing import Tuple


class AggregationFunction(Enum):
    """Aggregation functions for message passing."""
    SUM = 1
    AVG = 2
    MIN = 4
    MAX = 5


class MessagePassingLayer(nn.Module):
    """Single message passing layer."""
    
    def __init__(
        self, 
        num_nodes: int, 
        embedding_dim: int, 
        dim_proj: int, 
        dropout_prob: float, 
        skip_connections: bool, 
        aggregation_fn: AggregationFunction
    ):
        """
        Initialize message passing layer.
        
        Args:
            num_nodes: Number of nodes in the graph
            embedding_dim: Input embedding dimension
            dim_proj: Message projection dimension
            dropout_prob: Dropout probability
            skip_connections: Whether to use skip connections
            aggregation_fn: Aggregation function for messages
        """
        super().__init__()
        
        self.dim_proj = dim_proj
        self.num_nodes = num_nodes
        self.dropout_prob = dropout_prob
        self.skip_connections = skip_connections
        self.aggregation_fn = aggregation_fn
        
        # Message functions (learnable parameters)
        self.source_message_fn = mx.random.normal([1, embedding_dim, dim_proj])
        self.target_message_fn = mx.random.normal([1, embedding_dim, dim_proj])
        
        # Update function
        self.update_fn = nn.Linear(dim_proj, embedding_dim)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
    
    def __call__(self, connection_matrix: mx.array, node_embeddings: mx.array) -> mx.array:
        """
        Forward pass of message passing layer.
        
        Args:
            connection_matrix: Edge indices [2, num_edges]
            node_embeddings: Node embeddings [num_nodes, embedding_dim]
            
        Returns:
            Updated node embeddings [num_nodes, embedding_dim]
        """
        # Apply edge dropout (for regularization)
        if self.training:
            mask = mx.random.bernoulli(1 - self.dropout_prob, connection_matrix.shape)
            connection_matrix = connection_matrix * mask
        
        # Apply node dropout
        node_embeddings = self.dropout(node_embeddings)
        
        source_idx = connection_matrix[0]
        target_idx = connection_matrix[1]
        
        # Compute messages
        source_embeddings = node_embeddings @ self.source_message_fn
        target_embeddings = node_embeddings @ self.target_message_fn
        
        # Get edge-filtered embeddings
        filtered_source_embeddings = mx.take(source_embeddings, source_idx, axis=1)
        filtered_target_embeddings = mx.take(target_embeddings, target_idx, axis=1)
        
        # Combine source and target messages
        message = filtered_source_embeddings + filtered_target_embeddings
        message = self.relu(message)
        
        # Aggregate messages
        agg_message = mx.zeros([self.num_nodes, self.dim_proj])
        
        if self.aggregation_fn == AggregationFunction.SUM:
            agg_message = agg_message.at[target_idx].add(message)
        elif self.aggregation_fn == AggregationFunction.AVG:
            agg_message = agg_message.at[target_idx].add(message)
            denominator = mx.zeros([self.num_nodes, 1]).at[target_idx].add(1)
            agg_message = agg_message / mx.maximum(denominator, 1e-6)
        elif self.aggregation_fn == AggregationFunction.MAX:
            agg_message = agg_message.at[target_idx].maximum(message)
        elif self.aggregation_fn == AggregationFunction.MIN:
            agg_message = agg_message.at[target_idx].minimum(message)
        
        # Update node embeddings
        agg_message = self.dropout(agg_message)
        new_node_embeddings = self.update_fn(agg_message)
        new_node_embeddings = self.relu(new_node_embeddings)
        
        # Apply skip connection
        if self.skip_connections:
            new_node_embeddings = new_node_embeddings + node_embeddings
        
        return new_node_embeddings


class MPNN(nn.Module):
    """Message Passing Neural Network (MPNN) model."""
    
    def __init__(
        self, 
        num_nodes: int,
        embedding_dim: int, 
        dim_proj: int, 
        dropout_prob: float, 
        skip_connections: bool, 
        aggregation_fn: AggregationFunction, 
        num_mp_layers: int, 
        num_out_layers: int, 
        num_classes: int
    ):
        """
        Initialize MPNN model.
        
        Args:
            num_nodes: Number of nodes in the graph
            embedding_dim: Input embedding dimension
            dim_proj: Message projection dimension
            dropout_prob: Dropout probability
            skip_connections: Whether to use skip connections
            aggregation_fn: Aggregation function for messages
            num_mp_layers: Number of message passing layers
            num_out_layers: Number of output layers
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.dim_proj = dim_proj
        self.num_nodes = num_nodes
        self.dropout_prob = dropout_prob
        self.skip_connections = skip_connections
        self.aggregation_fn = aggregation_fn
        
        # Message passing layers
        self.mp_layers = [
            MessagePassingLayer(num_nodes, embedding_dim, dim_proj, dropout_prob, skip_connections, aggregation_fn)
            for _ in range(num_mp_layers)
        ]
        
        # Output layers
        self.out_layers = [
            nn.Linear(embedding_dim, embedding_dim)
            for _ in range(num_out_layers)
        ] + [nn.Linear(embedding_dim, num_classes)]
    
    def __call__(self, data: Tuple[mx.array, mx.array]) -> mx.array:
        """
        Forward pass of MPNN model.
        
        Args:
            data: Tuple of (node_embeddings, connection_matrix)
            
        Returns:
            Node predictions [num_nodes, num_classes]
        """
        node_embeddings, connection_matrix = data
        
        # Validate input dimensions
        if node_embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f'Incorrect node embedding size. Expected {self.embedding_dim}, '
                f'got {node_embeddings.shape[1]}'
            )
        
        # Apply message passing layers
        for mp_layer in self.mp_layers:
            node_embeddings = mp_layer(connection_matrix, node_embeddings)
        
        # Apply output layers
        for out_layer in self.out_layers:
            node_embeddings = out_layer(node_embeddings)
        
        return node_embeddings


# Aliases for backward compatibility
aggregation_fn = AggregationFunction
mp_layer = MessagePassingLayer
mpnn = MPNN
