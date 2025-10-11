"""
PPO agent for MPTCP path selection.

This module implements a Proximal Policy Optimization (PPO) agent for
selecting optimal paths in MPTCP. The agent provides probabilities for each
available path and learns from rewards received based on path performance.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .models import ActorCriticNetwork
from ..memory.replay_buffer import ReplayBuffer
from ..utils.normalizer import Normalizer
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import csv, os

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent for MPTCP path selection.
    
    This agent implements the approach from the process flow diagram:
    1. select_action: Called when a new flow arrives, returns path probabilities
    2. update: Called periodically (e.g., every 30s), updates based on path rewards
    
    Attributes:
        state_dim (int): Dimension of the state (path features)
        action_dim (int): Number of possible paths to choose from
        device (torch.device): Device to run the model on (CPU or GPU)
        actor_critic (ActorCriticNetwork): Neural network for policy and value functions
        optimizer (torch.optim.Optimizer): Optimizer for training the network
        normalizer (Normalizer): State feature normalizer
        buffer (ReplayBuffer): Experience replay buffer
        clip_epsilon (float): PPO clipping parameter
        gamma (float): Discount factor
        gae_lambda (float): GAE lambda parameter
        entropy_coef (float): Entropy coefficient for exploration
        value_loss_coef (float): Value loss coefficient
        max_grad_norm (float): Maximum gradient norm for clipping
        batch_size (int): Batch size for training
        update_epochs (int): Number of epochs to update per batch
        prev_path_stats (dict): Stores the previous path statistics for learning
    """
    
    def __init__(self, config):
        """
        Initialize the PPO agent with configuration parameters.
        
        Args:
            config (dict): Configuration parameters for the agent
        """
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.device = torch.device(config['device'])
        
        # Initialize actor-critic network
        self.actor_critic = ActorCriticNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=config['hidden_dim'],
            network_size=config['network_size']
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=config['learning_rate']
        )
        
        # Initialize normalizer for state features
        self.normalizer = Normalizer(self.state_dim)
        
        # Initialize replay buffer
        self.buffer = ReplayBuffer(
            state_dim=self.state_dim,
            action_dim=1,
            buffer_size=config['buffer_size'],
            batch_size=config['batch_size']
        )
        
        # PPO parameters
        self.clip_epsilon = config['clip_epsilon']
        self.gamma = config['gamma']
        self.gae_lambda = config['gae_lambda']
        self.entropy_coef = config['entropy_coef']
        self.value_loss_coef = config['value_loss_coef']
        self.max_grad_norm = config['max_grad_norm']
        self.batch_size = config['batch_size']
        self.update_epochs = config['update_epochs']
        
        # Store previous path stats for updates
        self.prev_path_stats = None

        # Exploration strategy smoothing 
        self.use_smoothed_probs = config['use_smoothed_probs']
        self.smoothed_probs = None
        self.alpha_smoothed_probs = config['alpha_smoothed_probs']

        ## logging training data
        self.timestamp = datetime.now().strftime("%d-%m_%H-%M")   # e.g. 05-05_09-07
        self.csv_path = f"RL/training_metrics/training_metrics_{self.timestamp}.csv"
        self.global_step = 0                                       # running counter
        self.writer = SummaryWriter(log_dir=f"RL/runs/ppo_agent/RUN_{self.timestamp}")      # one line
        
        # create header only the first time
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "step",
                "loss_total",
                "loss_policy",
                "loss_value",
                "policy_entropy",
                "reward_mean",
                "clip_fraction"
            ])
        
    def preprocess_state(self, path_stats):
        """
        Preprocess path statistics into a state vector.
        
        Args:
            path_stats (dict): Dictionary with path statistics 
                              {path_id: (flow_demand, ALU, MLU, delay_rank)}
        
        Returns:
            dict: Dictionary mapping path_ids to their normalized state vectors
            list: List of path_ids in the same order as the states
        """
        path_ids = list(path_stats.keys())
        states = {}
        
        for path_id, stats in path_stats.items():
            # Convert stats tuple to numpy array
            state = np.array(stats, dtype=np.float32)
            # Normalize state
            #normalized_state = self.normalizer.normalize(state)
            # without normalizer
            normalized_state = state
            states[path_id] = normalized_state
            
        return states, path_ids
        
    def select_action(self, path_stats):
        """
        Select paths based on their statistics using the trained policy.
        
        Args:
            path_stats (dict): Dictionary with path statistics 
                            {path_id: (flow_demand, ALU, MLU, delay_rank)}
        
        Returns:
            dict: Dictionary mapping path_ids to their selection probabilities
        """
        # smooth probabilities if enabled
        

        # Store current path stats for later updates
        self.prev_path_stats = path_stats
        
        # Extract path IDs and features
        path_ids = list(path_stats.keys())
        features = []
        
        for path_id in path_ids:
            # Convert to numpy array and normalize
            state = np.array(path_stats[path_id], dtype=np.float32)
            state = self.normalizer.normalize(state)
            features.append(state)
        
        # Convert to tensor - using numpy.array to avoid the warning and improve performance
        features_tensor = torch.FloatTensor(np.array(features)).to(self.device)
        
        # Get unnormalized scores (logits)
        with torch.no_grad():
            # Get features through the feature layers
            features = self.actor_critic.feature_layers(features_tensor)
            # Get logits from actor head
            logits = self.actor_critic.actor_head(features)
            
            # Apply softmax to get probabilities
            #probabilities = torch.nn.functional.softmax(logits, dim=0)
            # Convert to numpy array
            #probabilities = probabilities.cpu().numpy()
            #print(f"Probabilities: {probabilities}")
            #print(f"use_smoothed_probs: {self.use_smoothed_probs}")
            if self.use_smoothed_probs:
                # Convert logits to probabilities
                raw_probls = torch.nn.functional.softmax(logits, dim=0).cpu().numpy()
                # Apply exponential smoothing
                if self.smoothed_probs is None or self.smoothed_probs.shape != raw_probls.shape:
                    self.smoothed_probs = raw_probls
                else:
                    # print(f'size of smoothed_probs: {self.smoothed_probs.shape}')
                    # print(f'size of raw_probs: {raw_probls.shape}')
                    # print(f'global_step: {self.global_step}')
                    # print(f'batch_size: {self.batch_size}')
                    # print(f'buffer_size: {self.buffer.size()}')
                    self.smoothed_probs = (1 - self.alpha_smoothed_probs) * self.smoothed_probs + \
                                          self.alpha_smoothed_probs * raw_probls
                probabilities = self.smoothed_probs
                #print(f"Smoothed Probabilities: {probabilities}")
            else:
                # Apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(logits, dim=0)
                # Convert to numpy array
                probabilities = probabilities.cpu().numpy()

        
        # Create dictionary mapping path IDs to probabilities
        path_probs = {}
        for i, path_id in enumerate(path_ids):
            # Safely extract scalar value from the probability array
            if isinstance(probabilities[i], np.ndarray):
                # If it's a numpy array, ensure it's a scalar
                if probabilities[i].size == 1:
                    path_probs[path_id] = float(probabilities[i].item())
                else:
                    # If it's not size 1, take the first element
                    path_probs[path_id] = float(probabilities[i].flat[0])
            else:
                # If it's already a scalar
                path_probs[path_id] = float(probabilities[i])
        
        return path_probs
    
    def update(self, path_rewards, current_path_stats=None):
        """
        Update the agent based on rewards received for selected paths.
        This is called periodically (e.g., every 30 seconds).
        
        Args:
            path_rewards (dict): Dictionary mapping path_ids to their rewards
            current_path_stats (dict, optional): Current path statistics. If None,
                                                uses previous path statistics.
        """
        if self.prev_path_stats is None:
            if time.time() % 150 == 0:
                print("[PPO_Agent1]: Update called without previous path stats")
                print("[PPO_Agent1]: Warning: No previous path stats available for update")
            return
            
        # Use previous path stats if current stats not provided
        if current_path_stats is None:
            if time.time() % 150 == 0:
                print("[PPO_Agent1]: Update called without current path stats")
                print("[PPO_Agent1]: Warning: Using previous path stats for update")
            current_path_stats = self.prev_path_stats
        if time.time() % 150 == 0:
            print("[PPO_Agent1]: Update called with current path stats")
            print(f"[PPO_Agent1]: Current path stats: {current_path_stats}")

        # Check if we need to update action space due to new paths
        current_path_count = len(current_path_stats)
        if current_path_count > self.action_dim:
            # if time.time() % 150 == 0:
                # print(f"[PPO_Agent1]: Updating action space from {self.action_dim} to {current_path_count}")
            self.update_action_space(current_path_count)
            
        # Get previous states and path IDs
        prev_states, prev_path_ids = self.preprocess_state(self.prev_path_stats)
        current_states, _ = self.preprocess_state(current_path_stats)
        
        # Add experiences for paths that have rewards (i.e., were selected)
        for path_id, reward in path_rewards.items():
            if path_id not in prev_states:
                continue
                
            # Get state of the path
            state = prev_states[path_id]
            
            # Get next state if available
            next_state = current_states.get(path_id, None)
            if next_state is None:
                next_state = np.zeros_like(state)
                done = True
            else:
                done = False
                
            # Convert path ID to action index
            if path_id in prev_path_ids:
                action = prev_path_ids.index(path_id)
                # Skip if action is out of bounds for the current action space
                if action >= self.action_dim:
                    print(f"[PPO_Agent1]: Warning: Action {action} is out of bounds for action space {self.action_dim}")
                    continue
            else:
                continue
                
            # Store experience in buffer
            self.buffer.add(state, action, reward, next_state, done)
            
        # Update policy if buffer has enough samples
        # print("Agent1: Update called")
        # print(f"Buffer size: {self.buffer.size()}")
        # print(f"Batch size: {self.batch_size}")
        if self.buffer.size() >= self.batch_size:
            if time.time() % 150 == 0:
                print("[PPO_Agent1]: Updating policy...")
            # Update policy using PPO algorithm
            self._update_policy()
            
        # Store current path stats as previous for next update
        self.prev_path_stats = current_path_stats
        if time.time() % 150 == 0:
            print("[PPO_Agent1]: Update completed")

    def _update_policy(self):
        """
        Update the policy using PPO algorithm.
        """
        # Sample batch from buffer
        states, actions, rewards, next_states, dones = self.buffer.sample()
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Filter out actions that are out of bounds for the current action space
        valid_indices = []
        for i, action in enumerate(actions):
            if action < self.action_dim:
                valid_indices.append(i)
                
        if len(valid_indices) == 0:
            print("[PPO_Agent1]: Warning: No valid actions found for update")
            return
            
        # Use only valid data
        valid_indices = torch.tensor(valid_indices, device=self.device)
        states = states.index_select(0, valid_indices)
        #actions = torch.LongTensor([actions[i] for i in valid_indices]).to(self.device)
        # ...existing code...
        actions_np = np.array(actions)
        actions = torch.LongTensor(actions_np[valid_indices.cpu()].astype(int)).to(self.device)
        # ...existing code...
        rewards = rewards.index_select(0, valid_indices)
        next_states = next_states.index_select(0, valid_indices)
        dones = dones.index_select(0, valid_indices)
        
        # Reshape actions to have shape [batch_size, 1]
        actions = actions.view(-1, 1)
        
        # Get old action probabilities and values
        with torch.no_grad():
            old_action_probs, old_values = self.actor_critic(states)
            old_action_probs = old_action_probs.gather(1, actions)
            
            # Compute returns and advantages using GAE
            next_values = self.actor_critic.get_value(next_states)
            returns, advantages = self._compute_gae(rewards, old_values, next_values, dones)
        
        # Update for multiple epochs
        for _ in range(self.update_epochs):
            # Get current action probabilities and values
            action_probs, values = self.actor_critic(states)
            action_probs = action_probs.gather(1, actions)
            
            # Compute ratio and clipped loss
            ratio = action_probs / (old_action_probs + 1e-8)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Compute value loss
            value_loss = 0.5 * (returns - values).pow(2).mean()
            
            # Compute entropy bonus
            entropy = self.actor_critic.get_entropy(states).mean()
            
            # Compute total loss
            loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            
            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            
            # Increment global step
            self.global_step += 1
            #print(f"Global step: {self.global_step}")
            # Log losses and entropy
            # loss, policy_loss, value_loss and entropy are already in scope here
            self._log_scalars_csv(
                loss_total=loss.item(),
                loss_policy=policy_loss.item(),
                loss_value=value_loss.item(),
                policy_entropy=entropy.item(),
                reward_mean=rewards.mean().item(),
                clip_fraction=((ratio < 1-self.clip_epsilon) | (ratio > 1+self.clip_epsilon)).float().mean().item(),
            )
            
            # Log losses to TensorBoard
            self._log_scalars_tensor(
                loss_total=loss.item(),
                loss_policy=policy_loss.item(),
                loss_value=value_loss.item(),
                policy_entropy=entropy.item(),
                reward_mean=rewards.mean().item(),
                clip_fraction=((ratio < 1-self.clip_epsilon) | (ratio > 1+self.clip_epsilon)).float().mean().item(),
            )
            



    
    def _compute_gae(self, rewards, values, next_values, dones):
        """
        Compute returns and advantages using Generalized Advantage Estimation.
        
        Args:
            rewards (torch.Tensor): Batch of rewards
            values (torch.Tensor): Batch of value estimates
            next_values (torch.Tensor): Batch of next value estimates
            dones (torch.Tensor): Batch of done flags
            
        Returns:
            tuple: (returns, advantages)
        """
        batch_size = rewards.size(0)
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        last_gae_lam = 0
        
        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
                
            if dones[t]:
                delta = rewards[t] - values[t]
                last_gae_lam = delta
            else:
                delta = rewards[t] + self.gamma * next_value - values[t]
                last_gae_lam = delta + self.gamma * self.gae_lambda * last_gae_lam
                
            advantages[t] = last_gae_lam
            returns[t] = advantages[t] + values[t]
            
        return returns, advantages
    
    def save_model(self, path):
        """
        Save the model to the given path.
        
        Args:
            path (str): Path to save the model
        """
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'normalizer_state': self.normalizer.get_state(),
        }, path)
        
    def load_model(self, path):
        """
        Load the model from the given path.
        
        Args:
            path (str): Path to load the model from
        """
        checkpoint = torch.load(path)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.normalizer.set_state(checkpoint['normalizer_state'])
        
    def update_action_space(self, new_action_dim):
        """
        Update the action space when network topology changes.
        
        Args:
            new_action_dim (int): New number of possible paths
        """
        if new_action_dim == self.action_dim:
            return
            
        # Create new network with updated action dimension
        new_actor_critic = ActorCriticNetwork(
            state_dim=self.state_dim,
            action_dim=new_action_dim,
            hidden_dim=self.actor_critic.hidden_dim,
            network_size=self.actor_critic.network_size
        ).to(self.device)
        
        # Transfer shared parameters
        shared_dict = {k: v for k, v in self.actor_critic.state_dict().items() 
                      if 'actor_head' not in k}
        
        model_dict = new_actor_critic.state_dict()
        model_dict.update(shared_dict)
        new_actor_critic.load_state_dict(model_dict)
        
        # Replace old network and optimizer
        self.actor_critic = new_actor_critic
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=self.optimizer.param_groups[0]['lr']
        )
        
        # Update action dimension
        self.action_dim = new_action_dim
        
        # Clear buffer as actions may no longer be valid
        self.buffer.clear()

    def _log_scalars_tensor(self, **kwargs):
        """Write any number of scalars to TensorBoard."""
        for k, v in kwargs.items():
            self.writer.add_scalar(k, v, self.global_step)

    def _log_scalars(self, **kw):
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                self.global_step,
                kw["loss_total"],
                kw["loss_policy"],
                kw["loss_value"],
                kw["policy_entropy"],
                kw["reward_mean"],
                kw["clip_fraction"]
            ])


    def _log_scalars_csv(self, **metrics):
        """Append one training step to the CSV."""
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                self.global_step,
                metrics["loss_total"],
                metrics["loss_policy"],
                metrics["loss_value"],
                metrics["policy_entropy"],
                metrics["reward_mean"],
                metrics["clip_fraction"]
            ])