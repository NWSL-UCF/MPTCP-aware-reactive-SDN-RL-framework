"""
Replay buffer for PPO agent in MPTCP path selection.

This module provides a replay buffer for storing transitions between
path states, actions, and rewards for the PPO agent's learning process.
"""

import numpy as np
from collections import deque
import random

class ReplayBuffer:
    """
    Replay buffer for storing and sampling experience transitions.
    
    This buffer stores (state, action, reward, next_state, done) tuples
    for the PPO agent's learning process in MPTCP path selection.
    
    Attributes:
        buffer (deque): Double-ended queue to store transitions
        state_dim (int): Dimension of state features
        action_dim (int): Dimension of actions
        buffer_size (int): Maximum size of the buffer
        batch_size (int): Size of sampled batches
    """
    
    def __init__(self, state_dim, action_dim, buffer_size=10000, batch_size=64):
        """
        Initialize the replay buffer.
        
        Args:
            state_dim (int): Dimension of state features
            action_dim (int): Dimension of actions
            buffer_size (int): Maximum size of the buffer
            batch_size (int): Size of sampled batches
        """
        self.buffer = deque(maxlen=buffer_size)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state (np.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state
            done (bool): Whether the episode is done
        """
        # Handle case where next_state is None (e.g., path no longer exists)
        if next_state is None:
            next_state = np.zeros_like(state)
            done = True
            
        # Convert action to array if it's a scalar
        if isinstance(action, (int, float)):
            action = np.array([action])
            
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self):
        """
        Sample a batch of transitions from the buffer.
        
        Returns:
            tuple: (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, min(self.batch_size, len(self.buffer)))
        
        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards).reshape(-1, 1)
        next_states = np.array(next_states)
        dones = np.array(dones).reshape(-1, 1)
        
        return states, actions, rewards, next_states, dones
    
    def size(self):
        """
        Get the current size of the buffer.
        
        Returns:
            int: Number of transitions in the buffer
        """
        return len(self.buffer)
    
    def clear(self):
        """
        Clear the buffer.
        """
        self.buffer.clear()