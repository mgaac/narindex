"""
Graph Attention Network (GAT) implementation in MLX.
Based on "Graph Attention Networks" (Veličković et al., 2017)
arXiv:1710.10903v3
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple


class GATLayer(nn.Module):
    """Single Graph Attention Network layer."""
    
    def __init__(self, num_nodes: int, dim_proj: int, num_att_heads: int, dropout_prob: float):
        """
        Initialize GAT layer.
        
        Args:
            num_nodes: Number of nodes in the graph
            dim_proj: Projection dimension per attention head
            num_att_heads: Number of attention heads
            dropout_prob: Dropout probability
        """
        super().__init__()
        
        self.dim_proj = dim_proj
        self.num_nodes = num_nodes
        self.num_att_heads = num_att_heads
        
        # Attention score parameters (learnable)
        self.source_scores_fn = mx.random.normal([1, num_att_heads, dim_proj])
        self.target_scores_fn = mx.random.normal([1, num_att_heads, dim_proj])
        
        self.leaky_relu = nn.LeakyReLU(0.02)
        self.dropout = nn.Dropout(dropout_prob)
    
    def __call__(self, node_proj: mx.array, adjacency_matrix: mx.array) -> mx.array:
        """
        Forward pass of GAT layer.
        
        Args:
            node_proj: Node projections [num_nodes, num_heads * dim_proj]
            adjacency_matrix: Edge indices [2, num_edges]
            
        Returns:
            Updated node representations [num_nodes, num_heads * dim_proj]
        """
        source_idx = adjacency_matrix[0]
        target_idx = adjacency_matrix[1]
        
        # Reshape for multi-head attention
        node_proj = node_proj.reshape([-1, self.num_att_heads, self.dim_proj])
        
        # Compute attention scores
        source_scores = (node_proj * self.source_scores_fn).sum(axis=-1)
        target_scores = (node_proj * self.target_scores_fn).sum(axis=-1)
        
        # Get edge-filtered projections and scores
        edge_filtered_node_proj = mx.take(node_proj, source_idx, axis=0)
        edge_filtered_source_scores = mx.take(source_scores, source_idx, axis=0)
        edge_filtered_target_scores = mx.take(target_scores, target_idx, axis=0)
        
        # Compute edge attention scores
        edge_scores = self.leaky_relu(edge_filtered_source_scores + edge_filtered_target_scores)
        edge_scores = (edge_scores - edge_scores.max()).exp()
        
        # Compute softmax normalization
        softmax_denominator = mx.zeros([self.num_nodes, self.num_att_heads])
        softmax_denominator = softmax_denominator.at[target_idx].add(edge_scores)
        softmax_denominator = mx.take(softmax_denominator, target_idx, axis=0)
        
        # Apply attention and dropout
        attention_scores = edge_scores / (softmax_denominator + 1e-16)
        attention_scores = self.dropout(attention_scores)
        
        # Aggregate messages
        edge_filtered_node_proj = edge_filtered_node_proj * mx.expand_dims(attention_scores, axis=-1)
        new_node_proj = mx.zeros([self.num_nodes, self.num_att_heads, self.dim_proj])
        new_node_proj = new_node_proj.at[target_idx].add(edge_filtered_node_proj)
        new_node_proj = self.leaky_relu(new_node_proj)
        
        return new_node_proj.reshape((self.num_nodes, self.num_att_heads * self.dim_proj))


class GAT(nn.Module):
    """Graph Attention Network (GAT) model."""
    
    def __init__(
        self, 
        num_nodes: int,
        dim_embed: int, 
        dim_proj: int,
        num_att_heads: int,
        num_layers: int,
        skip_connections: bool,
        dropout_prob: float,
        num_out_layers: int,
        num_out_classes: int
    ):
        """
        Initialize GAT model.
        
        Args:
            num_nodes: Number of nodes in the graph
            dim_embed: Input embedding dimension
            dim_proj: Projection dimension per attention head
            num_att_heads: Number of attention heads
            num_layers: Number of GAT layers
            skip_connections: Whether to use skip connections
            dropout_prob: Dropout probability
            num_out_layers: Number of output layers
            num_out_classes: Number of output classes
        """
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_att_heads = num_att_heads
        self.dim_proj = dim_proj
        self.dim_embed = dim_embed
        self.skip_connections = skip_connections
        
        total_att_size = dim_proj * num_att_heads
        
        # Input projection
        self.embed_proj = nn.Linear(dim_embed, total_att_size)
        
        # GAT layers
        self.gat_layers = [
            GATLayer(num_nodes, dim_proj, num_att_heads, dropout_prob)
            for _ in range(num_layers)
        ]
        
        # Output layers
        self.out_layers = [
            nn.Linear(dim_proj, dim_proj)
            for _ in range(num_out_layers)
        ] + [nn.Linear(dim_proj, num_out_classes)]
        
        self.leaky_relu = nn.LeakyReLU(0.02)
        self.dropout = nn.Dropout(dropout_prob)
    
    def __call__(self, data: Tuple[mx.array, mx.array]) -> mx.array:
        """
        Forward pass of GAT model.
        
        Args:
            data: Tuple of (node_embeddings, adjacency_matrix)
            
        Returns:
            Node predictions [num_nodes, num_classes]
        """
        node_embeddings, adjacency_matrix = data
        
        # Validate input dimensions
        if node_embeddings.shape[1] != self.dim_embed:
            raise ValueError(
                f'Incorrect node embedding size. Expected {self.dim_embed}, '
                f'got {node_embeddings.shape[1]}'
            )
        
        # Input dropout and projection
        node_embeddings = self.dropout(node_embeddings)
        node_proj = self.embed_proj(node_embeddings)
        node_proj = self.dropout(node_proj)
        
        # Apply GAT layers
        for layer in self.gat_layers:
            new_node_proj = layer(node_proj, adjacency_matrix)
            if self.skip_connections:
                new_node_proj += node_proj
            node_proj = new_node_proj
        
        # Average across attention heads
        node_proj = node_proj.reshape(node_proj.shape[0], self.num_att_heads, self.dim_proj)
        node_proj = mx.mean(node_proj, axis=1)
        
        # Apply output layers
        for layer in self.out_layers:
            node_proj = layer(node_proj)
        
        return node_proj


# Alias for backward compatibility
gat = GAT
