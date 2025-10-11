# RL2/config.py
"""
Configuration for Agent2 Hierarchical GNN-PPO
"""

def get_config(network_size="small"):
    """Get configuration based on network size"""
    
    base_config = {
        # Agent parameters
        'device': 'auto',  # 'auto', 'cpu', or 'cuda'
        'seed': 42,
        'mask_invalid_actions': True,  # Enable masking for invalid actions
        'invalid_penalty': 0.2,  # Penalty for invalid actions
        # partial rewards
        'enable_partial_rewards': False,  # Enable partial reward learning
        'partial_reward_weights': {      # Weights for each level
            'switch': 0.3,
            'port': 0.2, 
            'flow': 0.3,
            'new_port': 0.2
        },
        # In config.py, fix the partial reward configuration:
        'partial_penalties': {           # Penalties for each level failure
            'switch_no_flows': -0.5,
            'port_no_flows': -0.3,
            'flow_selection_failed': -0.3,
            'new_port_selection_failed': -0.2
        },

        # GNN parameters OLD
        'gnn_type': 'graphsage',  # 'graphsage', 'gat', 'gcn'
        'gnn_layers': 3,
        'gnn_hidden_dim': 128,
        'gnn_output_dim': 64,
        'gnn_dropout': 0.15,

        # GNN parameters NEW
        #'gnn_type': 'graphsage',  # 'graphsage', 'gat', 'gcn'
        #'gnn_layers': 4,          # Number of GNN layers
        #'gnn_hidden_dim': 128,     # Hidden dimension for GNN layers
        #'gnn_output_dim': 64,     # Output dimension for GNN layers
        #'gnn_dropout': 0.3,       # Dropout rate for GNN layers

        # Hierarchical architecture
        'switch_features': 2,      # MLU, flow_count
        'port_features': 2,        # utilization, flow_count
        'flow_features': 2,        # rate, dst_dpid
        'new_port_features': 1,    # utilization
        
        # PPO parameters
        'learning_rate': 3e-4,
        'batch_size': 32,    # Batch size for training it was 32 16
        'n_epochs': 15,  # Number of epochs for each update
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.1,
        'value_loss_coef': 0.5,
        'entropy_coef'    : {
            'switch'  : 5e-3,
            'port'    : 5e-3,
            'flow'    : 5e-3,
            'new_port': 5e-3
        },
        'max_grad_norm': 0.5,

        # In RL2/config.py, add to the config dict:

        'no_op_rewards': {
            'excellent_state': 0.8,   # MLU < 0.5
            'good_state': 0.3,        # MLU < 0.7
            'poor_state': -0.2,       # MLU < 0.9
            'critical_state': -0.5,   # MLU >= 0.9
            'incomplete_action': -0.1,  # For flow/new_port No-Op

            # Flow level rewards (port utilization based)
            'flow_congested_port': 0.2,      # Port util > 0.8
            'flow_moderate_port': -0.05,     # Port util 0.6-0.8
            'flow_underutilized_port': -0.3, # Port util < 0.6
            
            # New port level
            'incomplete_action': -0.1    # For new_port No-Op
        },

        'no_op_enabled': True,  # Enable/disable No-Op globally
        'encourage_no_op': True,  # Encourage No-Op actions based on port utilization

        # Training parameters
        'update_interval': 30,    # 30 seconds
        'save_interval': 300,      # 5 minutes
        'eval_interval': 600,      # 10 minutes
        
        # Reward parameters
        'reward_type': 'normalized',  # 'simple', 'normalized', 'threshold'
        'mlu_thresholds': {
            'excellent': 0.5,
            'good': 0.7,
            'poor': 0.9
        },
        'reward_weights': {
            'mlu': 0.6,
            'improvement': 0.3,
            'stability': 0.1
        },
        
        # Memory and experience
        'memory_size': 10000,
        'min_memory_size': 5,  # Minimum size before training starts
        
        # Exploration
        'exploration_noise': 0.1,
        'exploration_decay': 0.995,
        'min_exploration': 0.01,
        
        # per–level PPO
        'min_mem': {
            'switch'  : 100,
            'port'    : 100,
            'flow'    : 100,
            'new_port': 100
        },
        'update_every': {
            'switch'  : 4,
            'port'    : 4,
            'flow'    : 6,      # usually noisier
            'new_port': 4
        },
        # learning-rates (4 optimisers)
        'lr': {
            'switch'  : 3e-4,
            'port'    : 3e-4,
            'flow'    : 1e-4,
            'new_port': 3e-4
        }

    }
    
    # Adjust for network size
    if network_size == "large":
        base_config.update({
            'gnn_hidden_dim': 128,
            'gnn_output_dim': 64,
            'batch_size': 64,
            'memory_size': 5000,
        })
    elif network_size == "medium":
        base_config.update({
            'gnn_hidden_dim': 96,
            'gnn_output_dim': 48,
            'batch_size': 48,
            'memory_size': 2000,
        })
    
    return base_config