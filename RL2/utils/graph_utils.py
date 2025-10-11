# RL2/utils/graph_utils.py
"""
Graph processing utilities
"""

import torch
import numpy as np
from collections import defaultdict


def extract_subgraph(graph_data, center_node, k_hop=2):
    """
    Extract k-hop subgraph around a center node
    
    Args:
        graph_data: Full graph data
        center_node: Center node ID
        k_hop: Number of hops
        
    Returns:
        dict: Subgraph data
    """
    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])
    
    # Find k-hop neighbors
    neighbors = {center_node}
    for _ in range(k_hop):
        new_neighbors = set()
        for edge in edges:
            if edge['from'] in neighbors:
                new_neighbors.add(edge['to'])
            if edge['to'] in neighbors:
                new_neighbors.add(edge['from'])
        neighbors.update(new_neighbors)
    
    # Extract subgraph
    subgraph_nodes = [n for n in nodes if n['id'] in neighbors]
    subgraph_edges = [
        e for e in edges 
        if e['from'] in neighbors and e['to'] in neighbors
    ]
    
    return {
        'nodes': subgraph_nodes,
        'edges': subgraph_edges
    }


def prepare_graph_data(nodes, edges, head_id):
    """
    Prepare graph data for GNN processing
    
    Args:
        nodes: List of nodes
        edges: List of edges
        head_id: Head to filter by
        
    Returns:
        tuple: (node_features, edge_index, node_ids)
    """
    # Filter nodes by head
    head_nodes = [n for n in nodes if n.get('head') == head_id]
    
    if not head_nodes:
        return None, None, None
    
    # Create node ID mapping
    node_ids = [n['id'] for n in head_nodes]
    node_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    
    # Extract features based on head type
    features = []
    for node in head_nodes:
        if head_id == 1:  # Switch
            features.append([node['data'].get('mlu', 0.0),
                             node['data'].get('flow_count', 0)
                             ])
        elif head_id == 2:  # Port
            features.append([
                node['data'].get('utilization', 0.0),
                node['data'].get('flow_count', 0)
            ])
        elif head_id == 3:  # Flow
            features.append([
                node['data'].get('rate', 0.0),
                float(node['data'].get('dst_dpid', 0))
            ])
        elif head_id == 4:  # New port
            features.append([node['data'].get('utilization', 0.0)])
    
    # Create edge index
    edge_list = []
    for edge in edges:
        if (edge['from'] in node_to_idx and 
            edge['to'] in node_to_idx):
            edge_list.append([
                node_to_idx[edge['from']], 
                node_to_idx[edge['to']]
            ])
    
    # Convert to tensors
    node_features = torch.FloatTensor(features)
    #edge_index = torch.LongTensor(edge_list).t() if edge_list else torch.LongTensor([[], []])
    edge_index = (torch.LongTensor(edge_list).t()
                if edge_list
                else torch.empty((2, 0), dtype=torch.long))

    return node_features, edge_index, node_ids


def calculate_graph_metrics(graph_data):
    """
    Calculate various graph metrics
    
    Args:
        graph_data: Graph data
        
    Returns:
        dict: Graph metrics
    """
    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])
    
    # Node counts by head
    head_counts = defaultdict(int)
    for node in nodes:
        head_counts[node.get('head', 0)] += 1
    
    # Edge counts by type
    edge_counts = defaultdict(int)
    for edge in edges:
        edge_counts[edge.get('edge_type', 'unknown')] += 1
    
    # Calculate density
    num_nodes = len(nodes)
    num_edges = len(edges)
    max_edges = num_nodes * (num_nodes - 1)
    density = num_edges / max_edges if max_edges > 0 else 0
    
    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'density': density,
        'head_counts': dict(head_counts),
        'edge_counts': dict(edge_counts)
    }


def get_node_features(node, head_id):
    """
    Extract features for a node based on its head type
    
    Args:
        node: Node data
        head_id: Head ID
        
    Returns:
        list: Feature vector
    """
    if head_id == 1:  # Switch
        return [
            node['data'].get('mlu', 0.0),
            node['data'].get('flow_count', 0),
            node['data'].get('port_count', 0)
        ]
    elif head_id == 2:  # Port
        return [
            node['data'].get('utilization', 0.0),
            node['data'].get('flow_count', 0),
            node['data'].get('bandwidth', 100),
            node['data'].get('delay', 0)
        ]
    elif head_id == 3:  # Flow
        return [
            node['data'].get('rate', 0.0),
            float(node['data'].get('dst_dpid', 0)),
            node['data'].get('priority', 1),
            node['data'].get('age', 0)
        ]
    elif head_id == 4:  # New port
        return [
            node['data'].get('utilization', 0.0),
            node['data'].get('available_bandwidth', 100)
        ]
    
    return []