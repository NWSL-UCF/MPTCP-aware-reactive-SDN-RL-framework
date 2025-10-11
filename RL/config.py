"""
Configuration settings for the PPO agent for MPTCP path selection.

This file contains hyperparameters and settings for the PPO agent,
organized by network size and use case.
"""

# Default configuration
DEFAULT_CONFIG = {
    # State and action dimensions
    'state_dim': 4,  # (flow_demand, ALU, MLU, delay_rank)
    'action_dim': 10,  # Will be dynamically updated based on available paths
    
    # Network architecture
    'hidden_dim': 64,
    'network_size': 'medium',  # small, medium, large
    'device': 'cpu',  # 'cpu' or 'cuda'
    
    # PPO parameters
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'entropy_coef': 0.01,
    'value_loss_coef': 0.5,
    'max_grad_norm': 0.5,
    
    # Training parameters
    'buffer_size': 10000,
    'batch_size': 64,
    'update_epochs': 6,
    
    # Exploration parameters
    'exploration_strategy': 'entropy',  # none, epsilon, entropy
    'epsilon_start': 1.0,
    'epsilon_end': 0.1,
    'epsilon_decay': 0.995, # Default: 0.995

    'use_smoothed_probs': False,  # Whether to use smoothed probabilities
    'alpha_smoothed_probs': 0.875,  # Smoothing factor for probability distribution
}

# Network size specific configurations
NETWORK_SIZE_CONFIGS = {
    'small': {
        'hidden_dim': 32,
        'learning_rate': 3e-4,
        'batch_size': 32,
        'buffer_size': 5000,
        'batch_size': 32,
        
    },
    'medium': {
        'hidden_dim': 64,
        'learning_rate': 1e-4,
        'batch_size': 64,
    },
    'large': {
        'hidden_dim': 128,
        'learning_rate': 5e-5,
        'batch_size': 128,
    }
}

def get_config(network_size='medium'):
    """
    Get configuration for specified network size.
    
    Args:
        network_size (str): Size of the network (small, medium, large)
        
    Returns:
        dict: Configuration parameters
    """
    config = DEFAULT_CONFIG.copy()
    
    if network_size in NETWORK_SIZE_CONFIGS:
        config.update(NETWORK_SIZE_CONFIGS[network_size])
        config['network_size'] = network_size
    
    return config