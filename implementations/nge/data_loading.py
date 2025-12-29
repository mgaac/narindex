"""Graph dataset generation and loading utilities for NGE.

Provides functions for generating synthetic graph datasets,
running graph algorithms (Bellman-Ford and BFS), and saving/loading datasets.
"""

import mlx.core as mx
import networkx as nx
import numpy as np

from math import inf
from typing import List, Tuple, Optional


def erdos_renyi_edge_matrix(num_nodes: int = 20, p: float = 0.2) -> mx.array:
    """Generate Erdős-Rényi random graph edge matrix.
    
    Args:
        num_nodes: Number of nodes in the graph
        p: Edge probability
        
    Returns:
        Edge matrix of shape (2, num_edges)
    """
    G = nx.erdos_renyi_graph(num_nodes, p)
    edges = list(G.edges())
    if not edges:
        return mx.array([])
    edge_matrix = mx.array(edges).T
    return edge_matrix


def barabasi_albert_edge_matrix(num_nodes: int = 20, m: int = 2) -> mx.array:
    """Generate Barabási-Albert preferential attachment graph edge matrix.
    
    Args:
        num_nodes: Number of nodes in the graph
        m: Number of edges to attach from a new node to existing nodes
        
    Returns:
        Edge matrix of shape (2, num_edges)
    """
    G = nx.barabasi_albert_graph(num_nodes, m)
    edges = list(G.edges())
    if not edges:
        return mx.array([])
    edge_matrix = mx.array(edges).T
    return edge_matrix


def add_self_loops(edge_matrix: mx.array, num_nodes: int) -> mx.array:
    """Add self loops to each node in the graph.
    
    Args:
        edge_matrix: Edge matrix of shape (2, num_edges)
        num_nodes: Number of nodes
        
    Returns:
        Edge matrix with self loops added
    """
    if num_nodes == 0:
        return edge_matrix
    
    self_loop_edges = [[i, i] for i in range(num_nodes)]
    
    if not self_loop_edges:
        return edge_matrix
    
    self_loops_array = mx.array(self_loop_edges).T
    
    if edge_matrix.size == 0:
        return self_loops_array
    else:
        return mx.concatenate([edge_matrix, self_loops_array], axis=1)


def make_bidirectional_edges(edge_matrix: mx.array) -> mx.array:
    """Convert directed edges to bidirectional by adding reverse edges.
    
    Self-loops (u,u) are NOT duplicated.
    
    Args:
        edge_matrix: Edge matrix of shape (2, num_edges) or (3, num_edges) with weights
        
    Returns:
        Bidirectional edge matrix
    """
    if edge_matrix.size == 0:
        return edge_matrix
    
    num_edges = edge_matrix.shape[1]
    bidirectional_edges = []
    
    for i in range(num_edges):
        u, v = int(edge_matrix[0, i]), int(edge_matrix[1, i])
        if edge_matrix.shape[0] > 2:  # Has weights
            w = edge_matrix[2, i]
            bidirectional_edges.append([u, v, w])
            if u != v:  # Not a self-loop: add reverse edge
                bidirectional_edges.append([v, u, w])
        else:
            bidirectional_edges.append([u, v])
            if u != v:  # Not a self-loop: add reverse edge
                bidirectional_edges.append([v, u])
    
    return mx.array(bidirectional_edges).T


def append_uniform_edge_weights(
    edge_matrix: mx.array,
    num_nodes: int,
    low: float = 0.2,
    high: float = 1.0
) -> mx.array:
    """Add uniform random edge weights and preprocess the graph.
    
    Adds self-loops, makes edges bidirectional, and appends random weights.
    
    Args:
        edge_matrix: Edge matrix of shape (2, num_edges)
        num_nodes: Number of nodes
        low: Minimum edge weight
        high: Maximum edge weight
        
    Returns:
        Edge matrix of shape (3, num_edges) with weights
    """
    if edge_matrix.size == 0 and num_nodes == 0:
        return edge_matrix
    
    # First add self loops (without weights initially)
    edge_matrix = add_self_loops(edge_matrix, num_nodes)
    
    # Then make edges bidirectional
    edge_matrix = make_bidirectional_edges(edge_matrix)
    
    num_edges = edge_matrix.shape[1]
    weights = mx.array(np.random.uniform(low, high, size=num_edges))
    # Stack as a new row: shape (3, num_edges)
    weighted_matrix = mx.concatenate([edge_matrix, weights.reshape(1, -1)], axis=0)
    return weighted_matrix


def bellman_ford_log(
    edges: mx.array,
    source: int,
    num_nodes: int,
) -> Tuple[List[List[float]], List[List[Optional[int]]]]:
    """Run Bellman-Ford algorithm and log each iteration.
    
    Args:
        edges: Edge matrix of shape (3, num_edges) with weights
        source: Source node index
        num_nodes: Number of nodes
        
    Returns:
        Tuple of (distance_log, predecessor_log)
    """
    if edges.size == 0:
        return [], []
    
    num_edges = edges.shape[1]
    
    # Build in-neighborhoods with weights
    in_neighbors: List[List[Tuple[int, float]]] = [[] for _ in range(num_nodes)]
    for k in range(num_edges):
        u = int(edges[0, k])
        v = int(edges[1, k])
        w = float(edges[2, k])
        in_neighbors[v].append((u, w))
    
    # Initialize
    distance: List[float] = [inf] * num_nodes
    predecessor: List[Optional[int]] = [None] * num_nodes
    distance[source] = 0.0
    predecessor[source] = source  # p_s = s for all iterations
    
    distance_log = [distance.copy()]
    predecessor_log = [predecessor.copy()]
    
    # Up to |V|-1 rounds
    for _ in range(num_nodes - 1):
        new_distance = distance.copy()
        new_predecessor = predecessor.copy()
        updated = False
        
        for i in range(num_nodes):
            if i == source:
                continue
            
            best_cost = inf
            best_pred = None
            
            # Best incoming relaxation candidate
            for j, w in in_neighbors[i]:
                cand = distance[j] + w
                if cand <= best_cost:
                    best_cost = cand
                    best_pred = j
            
            # Write only on genuine relaxation
            if best_cost < new_distance[i]:
                new_distance[i] = best_cost
                new_predecessor[i] = best_pred
                updated = True
        
        distance = new_distance
        predecessor = new_predecessor
        distance_log.append(distance.copy())
        predecessor_log.append(predecessor.copy())
        
        if not updated:
            break
    
    return distance_log, predecessor_log


def clean_bf_logs(log: List) -> mx.array:
    """Clean and normalize Bellman-Ford logs for training.
    
    Args:
        log: Either distance log or predecessor log
        
    Returns:
        Cleaned MLX array
    """
    if not log or not log[-1]:
        return mx.array(log)
    
    is_distance = isinstance(log[0][0], float)
    
    if is_distance:
        final = log[-1]
        finite = [x for x in final if x != float('inf')]
        S = (max(finite) + 1.0) if finite else 1.0
        
        norm_log = []
        for state in log:
            state_norm = [((S if x == float('inf') else x) / S) for x in state]
            norm_log.append(state_norm)
        return mx.array(norm_log, dtype=mx.float32)
    
    else:
        # Predecessors: replace None with -1 (sentinel for "no predecessor")
        cleaned = []
        for state in log:
            cleaned_state = [(-1 if x is None else x) for x in state]
            cleaned.append(cleaned_state)
        return mx.array(cleaned, dtype=mx.int32)


def bfs_log(
    edges: mx.array,
    source: int,
    num_nodes: int,
) -> List[List[int]]:
    """Run BFS algorithm and log each layer's reachability.
    
    Args:
        edges: Edge matrix of shape (3, num_edges)
        source: Source node index
        num_nodes: Number of nodes
        
    Returns:
        List of reachability states at each BFS layer
    """
    if edges.size == 0:
        return []
    
    # Build NetworkX graph for BFS (ignore weights for unweighted BFS)
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    for k in range(edges.shape[1]):
        u = int(edges[0, k])
        v = int(edges[1, k])
        G.add_edge(u, v)
    
    # Initialize reachability array: 1 if reachable, 0 if not
    reachable = [0] * num_nodes
    reachable[source] = 1
    
    reachability_log = [reachable.copy()]
    
    # Use NetworkX bfs_layers to get nodes layer by layer
    for layer in nx.bfs_layers(G, [source]):
        updated = False
        for node in layer:
            if not reachable[node]:
                reachable[node] = 1
                updated = True
        if updated:
            reachability_log.append(reachable.copy())
    
    return reachability_log


def generate_dataset(
    num_graphs: int = 100,
    num_nodes: int = 20,
    p: float = 0.2,
    m: int = 2
) -> List[dict]:
    """Generate a dataset of random graphs with algorithm execution traces.
    
    Args:
        num_graphs: Number of graphs to generate
        num_nodes: Number of nodes per graph
        p: Edge probability for Erdős-Rényi graphs
        m: Attachment parameter for Barabási-Albert graphs
        
    Returns:
        List of graph dictionaries
    """
    dataset = []
    
    for _ in range(num_graphs):
        # Randomly choose graph type
        if mx.random.uniform() < 0.5:
            edge_matrix = append_uniform_edge_weights(
                barabasi_albert_edge_matrix(num_nodes, m), num_nodes
            )
        else:
            edge_matrix = append_uniform_edge_weights(
                erdos_renyi_edge_matrix(num_nodes, p), num_nodes
            )
        
        source_node = mx.random.randint(0, num_nodes).item()
        
        # Bellman-Ford logs
        bf_distance, bf_predecessor = bellman_ford_log(edge_matrix, source_node, num_nodes)
        bf_distance = clean_bf_logs(bf_distance)
        bf_predecessor = clean_bf_logs(bf_predecessor)
        
        # BFS logs
        bfs_reachability = bfs_log(edge_matrix, source_node, num_nodes)
        bfs_reachability = mx.array(bfs_reachability)
        
        graph_dict = {
            "num_nodes": num_nodes,
            "edge_matrix": edge_matrix,
            "source_node": source_node,
            "bf_distance_targets": bf_distance,
            "bf_predecessor_targets": bf_predecessor,
            "bfs_state_targets": bfs_reachability,
        }
        dataset.append(graph_dict)
    
    return dataset


def save_dataset(dataset: List[dict], filename: str):
    """Save a dataset to disk using numpy's compressed format.
    
    Args:
        dataset: List of graph dictionaries
        filename: Output filename (will have .npz extension)
    """
    save_dict = {"num_graphs": len(dataset)}
    
    for i, graph_dict in enumerate(dataset):
        save_dict[f"num_nodes_{i}"] = graph_dict["num_nodes"]
        save_dict[f"edge_matrix_{i}"] = np.array(graph_dict["edge_matrix"])
        save_dict[f"source_node_{i}"] = graph_dict["source_node"]
        save_dict[f"bf_distance_targets_{i}"] = np.array(graph_dict["bf_distance_targets"])
        save_dict[f"bf_predecessor_targets_{i}"] = np.array(graph_dict["bf_predecessor_targets"])
        save_dict[f"bfs_state_targets_{i}"] = np.array(graph_dict["bfs_state_targets"])
    
    np.savez_compressed(filename, **save_dict)


def load_dataset(filename: str) -> List[dict]:
    """Load a dataset saved with save_dataset.
    
    Args:
        filename: Path to the .npz file
        
    Returns:
        List of graph dictionaries
    """
    loaded = np.load(filename, allow_pickle=True)
    num_graphs = int(loaded["num_graphs"])
    
    dataset = []
    for i in range(num_graphs):
        graph_dict = {
            "num_nodes": int(loaded[f"num_nodes_{i}"]),
            "edge_matrix": mx.array(loaded[f"edge_matrix_{i}"]),
            "source_node": int(loaded[f"source_node_{i}"]),
            "bf_distance_targets": mx.array(loaded[f"bf_distance_targets_{i}"]),
            "bf_predecessor_targets": mx.array(loaded[f"bf_predecessor_targets_{i}"]),
            "bfs_state_targets": mx.array(loaded[f"bfs_state_targets_{i}"]),
        }
        dataset.append(graph_dict)
    
    return dataset


if __name__ == "__main__":
    import os
    
    # Generate and save datasets when run directly
    print("Generating datasets...")
    
    # Determine output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    train_dataset = generate_dataset(1500, 20, 0.2, 2)
    val_dataset = generate_dataset(100, 20, 0.2, 2)
    test_dataset = generate_dataset(100, 20, 0.2, 2)
    
    save_dataset(train_dataset, os.path.join(data_dir, "train_dataset.npz"))
    save_dataset(val_dataset, os.path.join(data_dir, "val_dataset.npz"))
    save_dataset(test_dataset, os.path.join(data_dir, "test_dataset.npz"))
    
    print(f"Datasets saved to {data_dir}")

