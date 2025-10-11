"""
Neural network models for PPO agent in MPTCP path selection.

This module defines the neural network architectures used by the PPO agent
for path selection in MPTCP, including an Actor-Critic network that outputs
path selection probabilities and value estimates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorCriticNetwork(nn.Module):
    """
    Combined Actor-Critic network for the PPO algorithm.
    
    This network has shared feature extraction layers, followed by
    separate heads for the actor (policy) and critic (value function).
    
    Attributes:
        state_dim (int): Dimension of the state input
        action_dim (int): Dimension of the action output (number of paths)
        hidden_dim (int): Dimension of hidden layers
        network_size (str): Size of the network (small, medium, large)
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, network_size="medium"):
        """
        Initialize the Actor-Critic network.
        
        Args:
            state_dim (int): Dimension of the state input
            action_dim (int): Dimension of the action output (number of paths)
            hidden_dim (int): Dimension of hidden layers
            network_size (str): Size of the network (small, medium, large)
        """
        super(ActorCriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.network_size = network_size
        
        # Determine network architecture based on size
        if network_size == "small":
            self.feature_layers = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
        elif network_size == "medium":
            self.feature_layers = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU()
            )
        elif network_size == "large":
            self.feature_layers = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU()
            )
        else:
            raise ValueError(f"Unknown network size: {network_size}")
        
        # Actor head (policy network)
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        
        # Critic head (value network)
        self.critic_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state (torch.Tensor): Input state tensor
            
        Returns:
            tuple: (action_probs, value)
                - action_probs: Probabilities for each action
                - value: Estimated state value
        """
        # Extract features
        features = self.feature_layers(state)
        
        # Get action probabilities
        action_logits = self.actor_head(features)
        # Always apply softmax along the last dimension
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Get state value
        value = self.critic_head(features)
        
        return action_probs, value
    
    def get_value(self, state):
        """
        Get value estimate for a state.
        
        Args:
            state (torch.Tensor): Input state tensor
            
        Returns:
            torch.Tensor: Estimated state value
        """
        features = self.feature_layers(state)
        value = self.critic_head(features)
        return value
    
    def get_entropy(self, state):
        """
        Calculate entropy of the policy for a state.
        
        Args:
            state (torch.Tensor): Input state tensor
            
        Returns:
            torch.Tensor: Entropy of the policy
        """
        features = self.feature_layers(state)
        action_logits = self.actor_head(features)
        action_probs = F.softmax(action_logits, dim=-1)
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=-1)
        return entropy