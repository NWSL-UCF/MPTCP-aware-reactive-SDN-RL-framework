# RL2/models/ppo_networks.py
"""
PPO Actor-Critic Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalActor(nn.Module):
    """Actor network for hierarchical decision making"""

    def __init__(self, input_dim, hidden_dims=[256, 128]):
        super(HierarchicalActor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))  # Output layer for logits
        self.network = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # FIX: Use self.network instead of non-existent layers
        logits = self.network(x)
        
        # Return probabilities
        #return F.softmax(logits, dim=-1)  # last bug fix
        return logits  # For raw logits, use in loss calculation directly


class HierarchicalCritic(nn.Module):
    """Critic network for value estimation"""
    
    def __init__(self, input_dim, hidden_dims=[256, 128]):
        super(HierarchicalCritic, self).__init__()
        
        # Use flexible architecture like the actor
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))  # Output single value
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        value = self.network(x)
        return value


class AttentionActor(nn.Module):
    """Actor with attention mechanism for future use"""
    
    def __init__(self, input_dim, hidden_dim=128, num_heads=4):
        super(AttentionActor, self).__init__()
        
        # Ensure input_dim is divisible by num_heads
        if input_dim % num_heads != 0:
            raise ValueError(f"input_dim {input_dim} must be divisible by num_heads {num_heads}")
        
        self.attention = nn.MultiheadAttention(
            input_dim, num_heads, batch_first=True, dropout=0.1
        )
        
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x, mask=None):
        # Handle different input shapes
        original_shape = x.shape
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # Add batch dimension
        elif len(x.shape) == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dims
        
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm(x + attn_out)  # Residual connection
        
        # Handle output shape
        if len(original_shape) == 1:
            x = x.squeeze(0).squeeze(0)
        elif len(original_shape) == 2:
            x = x.squeeze(0)
        
        # MLP
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return F.softmax(logits, dim=-1)

class HierarchicalActorCritic(nn.Module):
    """Combined Actor-Critic for efficiency"""
    
    def __init__(self, input_dim, hidden_dims=[256, 128]):
        super(HierarchicalActorCritic, self).__init__()
        
        # Shared encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims[:-1]:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Separate heads
        final_dim = hidden_dims[-1]
        self.actor_head = nn.Sequential(
            nn.Linear(prev_dim, final_dim),
            nn.ReLU(),
            nn.Linear(final_dim, 1)
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(prev_dim, final_dim),
            nn.ReLU(),
            nn.Linear(final_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if 'actor' in str(m):
                    nn.init.orthogonal_(m.weight, gain=0.01)
                else:
                    nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        encoded = self.encoder(x)
        
        actor_logits = self.actor_head(encoded)
        critic_value = self.critic_head(encoded)
        
        return F.softmax(actor_logits, dim=-1), critic_value
    
    def get_action_and_value(self, x):
        """Convenience method for training"""
        probs, value = self.forward(x)
        return probs, value