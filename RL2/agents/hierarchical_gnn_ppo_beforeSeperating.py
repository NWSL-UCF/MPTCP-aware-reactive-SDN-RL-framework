# RL2/agents/hierarchical_gnn_ppo.py
"""
Hierarchical GNN-PPO Agent Implementation
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque
import logging
import setting
from ..models.gnn_models import GraphSAGEEncoder, GATEncoder
from ..models.ppo_networks import HierarchicalActor, HierarchicalCritic
from ..utils.reward_functions import calculate_reward
from ..utils.graph_utils import extract_subgraph, prepare_graph_data

# Add at the top with other imports For TensorBoard logging
from torch.utils.tensorboard import SummaryWriter
import csv
import os
from datetime import datetime
from typing import Optional, Tuple  # ← std-lib typing helpers

LOG = logging.getLogger(__name__)


class HierarchicalGNNPPOAgent:
    """
    Hierarchical GNN-PPO agent for multi-head graph based decision making
    """
    
    def __init__(self, config):
        self.config = config
        self.device = self._get_device(config['device'])
        
        # Set random seed
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        self.logging = setting.H_G_PPO_LOGGING

        # Initialize GNN encoders
        self._init_gnn_encoders()
        
        # Initialize PPO networks
        self._init_ppo_networks()
        
        # Initialize optimizers
        self._init_optimizers()
        
        # Memory and tracking
        #self.memory = deque(maxlen=config['memory_size'])
        self.memory = {
            'switch': deque(maxlen=config['memory_size']),
            'port': deque(maxlen=config['memory_size']),
            'flow': deque(maxlen=config['memory_size']),
            'new_port': deque(maxlen=config['memory_size'])
        }
        self.episode_rewards = []
        self.exploration_rate = config['exploration_noise']

        # Action validation tracking
        self.partial_actions = []
        self.enable_apply_mask = config.get('mask_invalid_actions', False)
        
        LOG.info(f"Hierarchical GNN-PPO Agent initialized on {self.device}")
        
        # Add metrics tracking initialization
        self.episode_rewards = []
        self.exploration_rate = config['exploration_noise']
        
        # Initialize TensorBoard and metrics directory
        self.run_timestamp = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
        self.metrics_dir = f"RL2/data/training_metrics/run_{self.run_timestamp}"
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.tb_writer = SummaryWriter(log_dir=os.path.join(self.metrics_dir, "tensorboard"))
        
        # Initialize CSV logging
        self.csv_path = os.path.join(self.metrics_dir, "training_metrics.csv")
        self._init_csv_logging()
        
        # Track training step
        self.training_step = 0
        self.reward_history = deque(maxlen=100)  # Keep last 100 rewards for averaging
        self.step = 0
        self.update_every = 4
        self.min_mem = 16
        
        LOG.info(f"Hierarchical GNN-PPO Agent initialized on {self.device}")
        LOG.info(f"Metrics will be saved to: {self.metrics_dir}")
    
    def _get_device(self, device_config):
        """Determine device to use"""
        if device_config == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device_config)
    
    def _init_gnn_encoders(self):
        """Initialize GNN encoders for each head"""
        gnn_type = self.config['gnn_type']
        
        if gnn_type == 'graphsage':
            encoder_class = GraphSAGEEncoder
        elif gnn_type == 'gat':
            encoder_class = GATEncoder
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        # Head 1: Switch encoder
        self.switch_encoder = encoder_class(
            input_dim=self.config['switch_features'],
            hidden_dim=self.config['gnn_hidden_dim'],
            output_dim=self.config['gnn_output_dim'],
            num_layers=self.config['gnn_layers'],
            dropout=self.config['gnn_dropout']
        ).to(self.device)
        
        # Head 2: Port encoder
        self.port_encoder = encoder_class(
            input_dim=self.config['port_features'],
            hidden_dim=self.config['gnn_hidden_dim'],
            output_dim=self.config['gnn_output_dim'],
            num_layers=self.config['gnn_layers'],
            dropout=self.config['gnn_dropout']
        ).to(self.device)
        
        # Head 3: Flow encoder
        self.flow_encoder = encoder_class(
            input_dim=self.config['flow_features'],
            hidden_dim=self.config['gnn_hidden_dim'],
            output_dim=self.config['gnn_output_dim'],
            num_layers=self.config['gnn_layers'],
            dropout=self.config['gnn_dropout']
        ).to(self.device)
        
        # Head 4: New port encoder
        self.new_port_encoder = encoder_class(
            input_dim=self.config['new_port_features'],
            hidden_dim=self.config['gnn_hidden_dim'],
            output_dim=self.config['gnn_output_dim'],
            num_layers=self.config['gnn_layers'],
            dropout=self.config['gnn_dropout']
        ).to(self.device)
    
    def _init_ppo_networks(self):
        """Initialize PPO actor-critic networks"""
        encoding_dim = self.config['gnn_output_dim']
        
        # Actors
        self.switch_actor = HierarchicalActor(encoding_dim).to(self.device)
        self.port_actor = HierarchicalActor(encoding_dim).to(self.device)
        self.flow_actor = HierarchicalActor(encoding_dim).to(self.device)
        self.new_port_actor = HierarchicalActor(encoding_dim).to(self.device)
        
        # Critics
        self.switch_critic = HierarchicalCritic(encoding_dim).to(self.device)
        self.port_critic = HierarchicalCritic(encoding_dim).to(self.device)
        self.flow_critic = HierarchicalCritic(encoding_dim).to(self.device)
        self.new_port_critic = HierarchicalCritic(encoding_dim).to(self.device)
    
    def _init_optimizers(self):
        """Initialize optimizers"""
        all_params = (
            list(self.switch_encoder.parameters()) +
            list(self.port_encoder.parameters()) +
            list(self.flow_encoder.parameters()) +
            list(self.new_port_encoder.parameters()) +
            list(self.switch_actor.parameters()) +
            list(self.port_actor.parameters()) +
            list(self.flow_actor.parameters()) +
            list(self.new_port_actor.parameters()) +
            list(self.switch_critic.parameters()) +
            list(self.port_critic.parameters()) +
            list(self.flow_critic.parameters()) +
            list(self.new_port_critic.parameters())
        )
        
        self.optimizer = torch.optim.Adam(
            all_params, 
            lr=self.config['learning_rate']
        )
    
    def _apply_mask(self, logits, invalid_mask):
        """Optionally zero-out logits where invalid_mask==True."""
        if not self.enable_apply_mask:
            return logits                       # masking disabled
        if invalid_mask.all():
            return None                         # special “no valid choice” signal
        logits = logits.clone()
        logits[invalid_mask] = -1e9
        return logits
    
    # ── hierarchical_gnn_ppo.py ──


    def _get_state_value(
            self,
            encoded: torch.Tensor,
            action: torch.Tensor,
            critic: nn.Module
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract the single embedding that corresponds to *action* and its value
        estimate, so every transition stored in self.memory has the same shape.

        Parameters
        ----------
        encoded : Tensor, shape [N, H]
            Full node-level embeddings produced by the encoder.
        action  : Tensor, 0-D (scalar index)
            Index selected by the actor (0 ≤ action < N).
        critic  : nn.Module
            Critic network for the current hierarchy level.

        Returns
        -------
        state : Tensor, shape [1, H]
            Only the chosen embedding, batched to keep PPO logic unchanged.
        value : Tensor, shape [1]
            Scalar value estimate for that exact state/action pair.
        """
        # keep_batch_dim ensures dimensions stay consistent across transitions
        state = encoded[action].unsqueeze(0).detach()
        raw = critic(encoded)[action]  # [N, 1] – all values for all actions
        value = raw.view(1).detach()  # keep_batch_dim
        return state, value

    # hierarchical_gnn_ppo.py  – place it just under _get_state_value()
    def _localise_action(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        After we collapse the chosen node to a single-row state we must also:
        • map its *global* index  → 0   (only one candidate now)
        • store a consistent old_log_prob (log 1 = 0)
        Returns
        -------
        local_action  : Tensor([0])         # shape (1,)
        local_logprob : Tensor([0.0])       # shape (1,)
        """
        local_action  = torch.zeros(1, dtype=torch.long, device=self.device)
        local_logprob = torch.zeros(1, dtype=torch.float32, device=self.device)
        return local_action, local_logprob
        # hierarchical_gnn_ppo.py  (inside HierarchicalGNNPPOAgent, just under
    # _localise_action, before any of the _select_* methods):

    # ──────────────────────────────────────────────────────────────────
    # Helper: run an encoder in inference mode even while the whole
    # agent stays in .train() for gradient updates.
    # ──────────────────────────────────────────────────────────────────
    def _encode_eval(self, encoder: nn.Module,
                     x: torch.Tensor,
                     edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward through *encoder* with BatchNorm & Dropout disabled.
        Automatically restores the original training / eval state.

        Returns
        -------
        Tensor  shape (N, H)  – node embeddings.
        """
        was_training = encoder.training
        encoder.eval()                       # disable BN + dropout
        with torch.no_grad():
            out = encoder(x, edge_index)
        if was_training:
            encoder.train()                  # restore original state
        return out.detach()




    def select_action(self, graph_data, network_state):
        """
        Hierarchically select action [S, F, P] with granular validation and partial rewards
        
        Args:
            graph_data: Multi-head graph data
            network_state: Current network state
            
        Returns:
            dict: Selected action with validation details and partial progress
        """
        try:
            action = {
                'valid': False,
                'switch': None,
                'port': None,
                'flow': None,
                'new_port': None,
                'validation': {},
                'partial_progress': [],  # Track which levels succeeded
                'failed_at_level': None,  # Track where failure occurred
                'partial_reward_eligible': self.config.get('enable_partial_rewards', False)
            }
            #LOG.info(f"[H_G_PPO-logging]: memory size: {len(self.memory)}")
            # Extract head data
            head_data = graph_data.get('head_data', {})
            pyg_data = graph_data.get('pyg_data', {})
            graph_viz = graph_data.get('graph_viz', {'nodes': [], 'edges': []})
            total_flows_in_system = 0
            

            if head_data.get(1, {}).get('nodes'):
                for switch_node in head_data[1]['nodes']:
                    total_flows_in_system += switch_node['data'].get('flow_count', 0)

            if total_flows_in_system == 0:
                LOG.warning("[H_G_PPO-logging]: No flows in the system, cannot select action")
                return {
                    'valid': False,
                    'switch': None,
                    'port': None,
                    'flow': None,
                    'new_port': None,
                    'validation': {
                        'error': 'No flows in the system'
                    },
                    'partial_progress': [],
                    'failed_at_level': 'no_flows',
                    'partial_reward_eligible': False, # no penalty - not agent's fault
                    'no_action_needed': True
                }

            if self.logging:
                print(f"[H_G_PPO-logging]: Starting hierarchical action selection")
            
            # Step 1: Select switch (S)
            switch_result = self._select_switch_with_validation(
                head_data.get(1, {}), graph_viz, pyg_data.get(1)
            )
            
            if switch_result['valid']:
                action['switch'] = switch_result['selection']
                action['partial_progress'].append('switch')
                action['validation']['switch'] = switch_result['validation']
                # ✅ NEW: Validate selected switch has flows
                selected_flow_count = switch_result['validation'].get('selected_flow_count', 0)
                if selected_flow_count == 0:
                    # Apply penalty immediately and stop
                    action['failed_at_level'] = 'switch'
                    action['validation']['switch']['error'] = 'Selected switch has no flows'
                    
                    if self.config.get('enable_partial_rewards', False):
                        # Store penalty for selecting invalid switch
                        #penalty = self.config.get('partial_penalties', {}).get('switch_no_flows', -0.5)
                        if switch_result.get('encoded_state') is not None:
                            LOG.info(f"[H_G_PPO-logging]: encoded_state: is valid")
                        self._store_partial_failure('switch', switch_result.get('encoded_state'))

                    LOG.info(f"[H_G_PPO-logging]: Selected switch {switch_result['selection']} has no flows, applying penalty")

                    return action  # ✅ Return immediately, don't proceed to port selection
                if self.logging:
                    LOG.info(f"[H_G_PPO-logging]: Selected switch: {switch_result['selection']}")

            else:
                action['failed_at_level'] = 'switch'
                action['validation']['switch'] = switch_result['validation']
                # Store partial transition for learning even on failure
                if self.config.get('enable_partial_rewards', False):
                    self._store_partial_failure('switch', switch_result.get('encoded_state'))
                return action
            
            # Step 2: Select port (hidden)
            port_result = self._select_port_with_validation(
                switch_result['selection'], graph_viz, pyg_data.get(2)
            )
            
            if port_result['valid']:
                action['port'] = port_result['selection']
                action['partial_progress'].append('port')
                action['validation']['port'] = port_result['validation']
                
                if self.logging:
                    print(f"[H_G_PPO-logging]: Selected port: {port_result['selection']}")
            else:
                action['failed_at_level'] = 'port'
                action['validation']['port'] = port_result['validation']
                if self.config.get('enable_partial_rewards', False):
                    if switch_result.get('encoded_state') is not None:
                            LOG.info(f"[H_G_PPO-logging]: encoded_state: is valid")
                    self._store_partial_failure('port', port_result.get('encoded_state'))
                return action
            
            # Step 3: Select flow (F)
            flow_result = self._select_flow(
                port_result['selection'], graph_viz, pyg_data.get(3)
            )
            
            if flow_result['valid']:
                action['flow'] = flow_result['selection']
                action['partial_progress'].append('flow')
                action['validation']['flow'] = flow_result['validation']
                
                if self.logging:
                    print(f"[H_G_PPO-logging]: Selected flow: {flow_result['selection']}")
            else:
                action['failed_at_level'] = 'flow'
                action['validation']['flow'] = flow_result['validation']
                if self.config.get('enable_partial_rewards', False):
                    if switch_result.get('encoded_state') is not None:
                            LOG.info(f"[H_G_PPO-logging]: encoded_state: is valid")
                    self._store_partial_failure('flow', flow_result.get('encoded_state'))
                return action
            
            # Step 4: Select new port (P)
            new_port_result = self._select_new_port(
                flow_result['selection'], graph_viz, pyg_data.get(4)
            )
            
            if new_port_result['valid']:
                action['new_port'] = new_port_result['selection']
                action['partial_progress'].append('new_port')
                action['validation']['new_port'] = new_port_result['validation']
                action['valid'] = True  # Full action is valid
            else:
                action['failed_at_level'] = 'new_port'
                action['validation']['new_port'] = new_port_result['validation']
                if self.config.get('enable_partial_rewards', False):
                    if switch_result.get('encoded_state') is not None:
                            LOG.info(f"[H_G_PPO-logging]: encoded_state: is valid")
                    self._store_partial_failure('new_port', new_port_result.get('encoded_state'))
                return action
            
            return action
                
        except Exception as e:
            LOG.error(f"Error in action selection: {e}")
            return {
                'valid': False, 
                'failed_at_level': 'exception',
                'partial_progress': [],
                'error': str(e)
            }

    def _select_port_with_validation(self, switch_id, graph_viz, port_data=None):
        """Select a port with proper masking and validation"""
        try:
            result = {
                'valid': False,
                'selection': None,
                'validation': {},
                'encoded_state': None
            }
            
            # Get ports connected to this switch
            port_nodes = []
            for edge in graph_viz['edges']:
                edge_type_attr = edge.get('edge_type') or edge.get('type')
                if (edge['from'] == switch_id and
                        edge_type_attr in ('switch_link', 'switch_to_port')):
                    port_id = edge['to']
                    for node in graph_viz['nodes']:
                        if node['id'] == port_id and node['type'] == 'port':
                            port_nodes.append(node)
                            break
            
            if not port_nodes:
                result['validation']['error'] = 'No port nodes found for switch'
                return result
            
            # Prepare features and validate flow counts
            features = []
            node_ids = []
            valid_ports = []
            
            for node in port_nodes:
                util = node['data'].get('utilization', 0.0)
                flow_count = node['data'].get('flow_count', 0)
                features.append([util, flow_count])
                node_ids.append(node['id'])
                valid_ports.append(flow_count > 0)
            
            # Check if any port has flows
            if not any(valid_ports):
                result['validation']['error'] = 'No ports with flows available'
                result['validation']['flow_counts'] = [f[1] for f in features]
                return result
            
            # Convert to tensor and encode
            x = torch.FloatTensor(features).to(self.device)
            
            if port_data is not None:
                edge_index = port_data.edge_index.to(self.device)
            else:
                # Create self-loop edges as fallback
                num_ports = len(features)
                edge_index = torch.LongTensor([[i, i] for i in range(num_ports)]).t().to(self.device)
            
            with torch.no_grad():
                #encoded = self.port_encoder(x, edge_index)
                encoded = self._encode_eval(self.port_encoder, x, edge_index)
                result['encoded_state'] = encoded
                
                logits = self.port_actor(encoded).squeeze(-1)
                
                # Apply proper masking for ports with no flows
                flow_tensor = torch.tensor([f[1] for f in features], device=self.device)
                invalid_mask = (flow_tensor == 0)
                
                if self.config.get('mask_invalid_actions', True):
                    logits = self._apply_mask(logits, invalid_mask)
                    if logits is None:  # All ports invalid
                        result['validation']['error'] = 'All ports masked due to no flows'
                        return result
                
                # Add exploration noise
                if self.exploration_rate > self.config['min_exploration']:
                    logits = logits + torch.randn_like(logits) * self.exploration_rate
                
                probs = F.softmax(logits, dim=0)
                dist = Categorical(probs)
                action = dist.sample()
                
                # Get value for the selected port
                '''value = self.port_critic(encoded)[action].unsqueeze(0)
                
                # Store transition
                self._store_transition(
                    'port',
                    encoded[action].unsqueeze(0),
                    action,
                    dist.log_prob(action),
                    value
                )'''

                state, value = self._get_state_value(encoded, action, self.port_critic)
                #self._store_transition('port', state, action, dist.log_prob(action), value)
                loacl_act, local_lp = self._localise_action()
                self._store_transition(
                    'port',
                    state,
                    loacl_act,
                    local_lp,
                    value
                )

                selected_idx = action.item()
                result['valid'] = True
                result['selection'] = node_ids[selected_idx]
                result['validation']['selected_flow_count'] = features[selected_idx][1]
                result['validation']['total_ports'] = len(port_nodes)
                result['validation']['valid_ports'] = sum(valid_ports)
            
            return result
            
        except Exception as e:
            return {
                'valid': False,
                'selection': None,
                'validation': {'error': f'Exception in port selection: {e}'},
                'encoded_state': None
            }
    def select_action1(self, graph_data, network_state):
        """
        Hierarchically select action [S, F, P] with granular validation
        
        Args:
            graph_data: Multi-head graph data
            network_state: Current network state
            
        Returns:
            dict: Selected action with validation details
        """
        try:
            action = {
                'valid': False,
                'switch': None,
                'port': None,
                'flow': None,
                'new_port': None,
                'validation': {},
                'partial_reward_eligible': False,
                }
            
            # Extract head data
            head_data = graph_data.get('head_data', {})
            pyg_data = graph_data.get('pyg_data', {})
            graph_viz = graph_data.get('graph_viz', {'nodes': [], 'edges': []})
            

            if self.logging:
                print(f"[H_G_PPO-logging]: Graph visualization: {graph_viz}")
                print(f"[H_G_PPO-logging]: Head data: {head_data}")
                print(f"[H_G_PPO-logging]: PyG data: {pyg_data}")
            # Step 1: Select switch (S)
            switch_id = self._select_switch(head_data.get(1, {}), graph_viz,
                                pyg_data.get(1))
            if not switch_id:
                LOG.warning("[hierarical_GNN_PPO] WARNING: No valid switch selected")
                return {'valid': False}
            if self.logging:
                print(f"[H_G_PPO-logging]: Selected switch: {switch_id}")
            # Step 2: Select port (hidden)
            port_id = self._select_port(switch_id, graph_viz,
                                        pyg_data.get(2))
            if not port_id:
                LOG.warning("[hierarical_GNN_PPO] WARNING: No valid port selected")
                return {'valid': False}
            if self.logging:
                print(f"[H_G_PPO-logging]: Selected port: {port_id}")
            # Step 3: Select flow (F)
            flow_id = self._select_flow(port_id, graph_viz,
                                        pyg_data.get(3))
            if not flow_id:
                LOG.warning("[hierarical_GNN_PPO] WARNING: No valid flow selected")
                return {'valid': False}
            if self.logging:
                print(f"[H_G_PPO-logging]: Selected flow: {flow_id}")

            # Step 4: Select new port (P)
            new_port_id = self._select_new_port(flow_id, graph_viz,
                                                pyg_data.get(4))
            if not new_port_id:
                LOG.warning("[hierarical_GNN_PPO] WARNING: No valid new port selected")
                return {'valid': False}
            
            return {
                'valid': True,
                'switch': switch_id,
                'flow': flow_id,
                'new_port': new_port_id,
                '_hidden_port': port_id,  # For internal tracking
                'exploration': np.random.random() < self.exploration_rate
            }
            
        except Exception as e:
            LOG.error(f"Error in action selection: {e}")
            return {'valid': False}

    def _select_switch(self, head1_data, graph_viz, switch_data=None):
        """Select switch based on MLU and flow count"""
        try:
            # Get switch nodes
            if switch_data is not None:
                x = switch_data.x.to(self.device)
                edge_index = switch_data.edge_index.to(self.device)
                node_ids = getattr(switch_data, 'node_ids', list(range(x.size(0))))
                with torch.no_grad():
                    encoded = self.switch_encoder(x, edge_index)
                    probs = self.switch_actor(encoded)
                    if self.exploration_rate > self.config['min_exploration']:
                        noise = torch.randn_like(probs) * self.exploration_rate
                        probs = F.softmax(probs + noise, dim=-1)
                    if len(probs.shape) > 1:
                        probs = probs.squeeze()
                    dist = Categorical(probs)
                    action = dist.sample()
                    value = self.switch_critic(encoded)
                    self._store_transition('switch', encoded, action,
                                           dist.log_prob(action), value)
                return node_ids[action.item()]
            
            # Fallback to visualization data
            if self.logging:
                print(f"[H_G_PPO-logging]: Graph visualization: {graph_viz}")
            switch_nodes = [
                n for n in graph_viz['nodes'] 
                if n.get('type') == 'switch' or n.get('head') == 1
            ]
            if self.logging:
                print(f"[H_G_PPO-logging]: Switch nodes: {switch_nodes}")
            
            if not switch_nodes:
                LOG.warning("[hierarchical_GNN_PPO] Warning: No switch nodes found")
                return None
            
            # Prepare features
            features = []
            node_ids = []
            
            for node in switch_nodes:

                mlu = 0.0  # Default value
                flow_cnt = 0  # Default value
                # Try to get MLU from head1_data if available
                if head1_data and 'nodes' in head1_data:
                    for head_node in head1_data['nodes']:
                        if head_node.get('dpid') == node.get('dpid'):
                            mlu = head_node.get('mlu', 0.0)
                            flow_cnt = head_node.get('flow_count', 0)
                            break
                
                #features.append([mlu])
                features.append([mlu, flow_cnt])
                node_ids.append(node['id'])

            
            # Convert to tensor
            x = torch.FloatTensor(features).to(self.device)



            # Encode with GNN
            edge_index = self._extract_edges(graph_viz, 'switch_link', node_ids)

            with torch.no_grad():
                '''encoded = self.switch_encoder(x, edge_index)
                
                # Get action probabilities
                probs = self.switch_actor(encoded)
                
                # Add exploration noise
                if self.exploration_rate > self.config['min_exploration']:
                    noise = torch.randn_like(probs) * self.exploration_rate
                    probs = F.softmax(probs + noise, dim=-1)'''
                encoded = self.switch_encoder(x, edge_index)
                logits = self.switch_actor(encoded).squeeze(-1)

                # Mask switches with no flows
                flow_tensor = torch.tensor([f[1] for f in features],
                                             device=self.device)
                
                invalid_mask = (flow_tensor == 0)
                #logits[invalid_mask] = float('-1e9')  # Mask out no-flow switches
                logits = self._apply_mask(logits, invalid_mask)

                if self.exploration_rate > self.config['min_exploration']:
                    logits = logits + torch.randn_like(logits) * self.exploration_rate

                probs = F.softmax(logits, dim=-1)


                
                # Sample action
                if len(probs.shape) > 1:
                    probs = probs.squeeze()
                    
                dist = Categorical(probs)
                action = dist.sample()
                
                # Get value estimate
                value = self.switch_critic(encoded)
                
                # Store in memory
                self._store_transition('switch', encoded, action,
                                       dist.log_prob(action), value)
            
            return node_ids[action.item()]
            
        except Exception as e:
            LOG.error(f"Error selecting switch: {e}")
            return None

    def _select_port(self, switch_id, graph_viz, port_data=None):
        """Select a port for the chosen switch.

        The method first tries to use the PPO actor/critic pair when
        PyG data for ports is available. When no graph data is present it
        falls back to a simple heuristic based on utilization and flow
        count.
        """
        
        try:
            # Get ports connected to this switch
            port_nodes = []
            
            for edge in graph_viz['edges']:
                edge_type_attr = edge.get('edge_type') or edge.get('type')
                if (edge['from'] == switch_id and
                        edge_type_attr in ('switch_link', 'switch_to_port')):
                    port_id = edge['to']
                    # Find port node
                    for node in graph_viz['nodes']:
                        if node['id'] == port_id and node['type'] == 'port':
                            port_nodes.append(node)
                            break
            
            if not port_nodes:
                return None
            
            # Simple selection based on utilization and flow count
            # This is hidden from the user


            node_ids = [n['id'] for n in port_nodes]

            if port_data is not None:
                # Extract features from PyG data
                features = []
                for node_id in node_ids:

                    util = 0.0  # Default value
                    flow_count = 0  # Default value
                    for node in port_nodes:
                        if node['id'] == node_id:
                            util = node['data'].get('utilization', 0.0)
                            flow_count = node['data'].get('flow_count', 0)
                            break
                    features.append([util, flow_count])
                # Convert to tensor
                x = torch.FloatTensor(features).to(self.device)
                edge_index = port_data.edge_index.to(self.device)

                with torch.no_grad():
                    # Encode with GNN
                    encoded = self.port_encoder(x, edge_index)
                    '''
                    # Get action probabilities
                    logits = self.port_actor(encoded)
                    
                    # Add exploration noise
                    if self.exploration_rate > self.config['min_exploration']:
                        noise = torch.randn_like(logits) * self.exploration_rate
                        logits = logits + noise

                    probs = F.softmax(logits, dim=-1)

                    if len(probs.shape) > 1:
                        probs = probs.squeeze()'''
                    # ---- logits, mask --------------------------------------------------
                    logits = self.port_actor(encoded).squeeze(-1)           # [N_ports]
                    flow_tensor = torch.tensor([f[1] for f in features],
                                            device=self.device, dtype=torch.long)
                    invalid_mask = (flow_tensor == 0)
                    if invalid_mask.all():          # no valid port → fallback
                        return None
                    #logits[invalid_mask] = -1e9
                    logits = self._apply_mask(logits, invalid_mask)  # Mask out no-flow ports

                    # ---- exploration ---------------------------------------------------
                    if self.exploration_rate > self.config['min_exploration']:
                        logits = logits + torch.randn_like(logits) * self.exploration_rate

                    probs = F.softmax(logits, dim=0)

                    # ---- sample & value ------------------------------------------------
                    dist   = Categorical(probs)
                    action = dist.sample()
                    value  = self.port_critic(encoded)[action]              # scalar

                    # ---- store transition ---------------------------------------------
                    self._store_transition(
                        'port',
                        encoded[action].unsqueeze(0),                       # 1×H
                        action,
                        dist.log_prob(action),
                        value,
                    )

                    return node_ids[action.item()]

                    logits = self.port_actor(encoded).squeeze(-1) # shape -> [num_ports]

                    flow_tensor = torch.tensor([f[1] for f in features],
                                             device=self.device)
                    invalid_mask = (flow_tensor == 0)
                    logits[invalid_mask] = float('-1e9')  # Mask out no-flow ports
                    if self.exploration_rate > self.config['min_exploration']:
                        logits = logits + torch.randn_like(logits) * self.exploration_rate
                    probs = F.softmax(logits, dim=0) # shape -> [num_ports]
                    
                    dist = Categorical(probs)
                    action = dist.sample()
                    
                    # Get value estimate
                    value = self.port_critic(encoded)
                    
                    # Store in memory
                    self._store_transition('port', encoded, action,
                                           dist.log_prob(action), value)
                    return node_ids[action.item()]


                '''x = port_data.x.to(self.device)
                edge_index = port_data.edge_index.to(self.device)
                all_node_ids = getattr(port_data, 'node_ids', list(range(x.size(0))))
                node_to_idx = {nid: idx for idx, nid in enumerate(all_node_ids)}
                indices = [node_to_idx[nid] for nid in node_ids if nid in node_to_idx]

                if indices:
                    with torch.no_grad():
                        encoded_all = self.port_encoder(x, edge_index)
                        encoded = encoded_all[indices]
                        probs = self.port_actor(encoded)
                        if self.exploration_rate > self.config['min_exploration']:
                            noise = torch.randn_like(probs) * self.exploration_rate
                            probs = F.softmax(probs + noise, dim=-1)
                        if len(probs.shape) > 1:
                            probs = probs.squeeze()
                        dist = Categorical(probs)
                        action = dist.sample()
                        value = self.port_critic(encoded)
                        self._store_transition('port', encoded, action,
                                               dist.log_prob(action), value)
                    return node_ids[action.item()]'''

            # Fallback to simple heuristic

            best_port = None
            best_score = float('inf')
            
            for port in port_nodes:
                util = port['data'].get('utilization', 0.0)
                flow_count = port['data'].get('flow_count', 0)
                
                # Score: lower is better
                score = util + 0.1 * flow_count
                
                if score < best_score:
                    best_score = score
                    best_port = port['id']
            
            return best_port
            
        except Exception as e:
            LOG.error(f"Error selecting port: {e}")
            return None

    def _select_flow(self, port_id, graph_viz, flow_data=None):
        """Select flow from port"""
        try:
            result = {
                'valid': False,
                'selection': None,
                'validation': {},
                'encoded_state': None
            }
            # Get flows connected to this port
            flow_nodes = []
            
            for edge in graph_viz['edges']:
                if (edge['from'] == port_id and 
                    edge['edge_type'] == 'port_to_flow'):
                    flow_id = edge['to']
                    # Find flow node
                    for node in graph_viz['nodes']:
                        if node['id'] == flow_id and node['type'] == 'flow':
                            flow_nodes.append(node)
                            break
            
            if not flow_nodes:
                return result
            
            # Prepare features
            features = []
            node_ids = []
            
            for node in flow_nodes:
                rate = node['data'].get('rate', 0.0)
                dst_dpid = float(node['data'].get('dst_dpid', 0))
                features.append([rate, dst_dpid])
                node_ids.append(node['id'])
            
            # Convert to tensor
            x = torch.FloatTensor(features).to(self.device)
            
            if flow_data is not None:
                edge_index = flow_data.edge_index.to(self.device)
            else:
                # Create self-loop edges as fallback
                num_flows = len(features)
                edge_index = torch.LongTensor([[i, i] for i in range(num_flows)]).t().to(self.device)


            # Simple encoding (can be enhanced with local GNN)
            with torch.no_grad():
                # For now, just use first layer of encoder
                #encoded = self.flow_encoder(x, edge_index)
                encoded = self._encode_eval(self.flow_encoder, x, edge_index)
                result['encoded_state'] = encoded
                
                # Get action probabilities
                logits = self.flow_actor(encoded).squeeze(-1)

                # Add exploration noise
                if self.exploration_rate > self.config['min_exploration']:
                    noise = torch.randn_like(logits) * self.exploration_rate
                    logits = logits + noise
                
                probs = F.softmax(logits, dim=0)
                dist = Categorical(probs)
                action = dist.sample()
                
                '''# Get value estimate
                value = self.flow_critic(encoded)[action].unsqueeze(0)
                
                # Store in memory
                self._store_transition(
                    'flow',
                     encoded[action].unsqueeze(0), 
                     action,
                    dist.log_prob(action), 
                    value
                )'''
                state, value = self._get_state_value(encoded, action, self.flow_critic)
                loacl_act, local_lp = self._localise_action()
                self._store_transition(
                    'flow',
                    state,
                    loacl_act,
                    local_lp,
                    value
                )

                selected_idx = action.item()
                result['valid'] = True
                result['selection'] = node_ids[selected_idx]
                result['validation']['selected_flow_rate'] = features[selected_idx][0]
                result['validation']['dst_dpid'] = features[selected_idx][1]
            
            return result
            
        except Exception as e:
            LOG.error(f"Error selecting flow: {e}")
            return {
                'valid': False,
                'selection': None,
                'validation': {'error': f'Exception in flow selection: {e}'},
                'encoded_state': None
            }

    def _select_new_port(self, flow_id, graph_viz, new_port_data=None):
        """Select new port for flow"""
        try:
            result = {
                'valid': False,
                'selection': None,
                'validation': {},
                'encoded_state': None
            }
            # Get new ports connected to this flow
            port_nodes = []
            
            for edge in graph_viz['edges']:
                if (edge['from'] == flow_id and 
                    edge['edge_type'] == 'flow_to_new_port'):
                    port_id = edge['to']
                    # Find new port node
                    for node in graph_viz['nodes']:
                        if node['id'] == port_id and node['type'] == 'new_port':
                            port_nodes.append(node)
                            break
            
            if not port_nodes:
                return result
            
            # Prepare features
            features = []
            node_ids = []
            
            for node in port_nodes:
                util = node['data'].get('utilization', 0.0)
                features.append([util])
                node_ids.append(node['id'])
            
            # Convert to tensor
            x = torch.FloatTensor(features).to(self.device)

            if new_port_data is not None:
                edge_index = new_port_data.edge_index.to(self.device)
            else:
                # Create self-loop edges as fallback
                num_ports = len(features)
                edge_index = torch.LongTensor([[i, i] for i in range(num_ports)]).t().to(self.device)
            
            with torch.no_grad():
                # Simple encoding
                #encoded = self.new_port_encoder(x, edge_index)
                encoded = self._encode_eval(self.new_port_encoder, x, edge_index)
                result['encoded_state'] = encoded
                
                
                # Get action probabilities
                logits = self.new_port_actor(encoded).squeeze(-1)
                # Add exploration noise
                if self.exploration_rate > self.config['min_exploration']:
                    noise = torch.randn_like(logits) * self.exploration_rate
                    logits = logits + noise
                probs = F.softmax(logits, dim=0)
                dist = Categorical(probs)
                action = dist.sample()

                # Get value estimate
                #value = self.new_port_critic(encoded)[action].unsqueeze(0)

                # Store in memory
                '''self._store_transition(
                    'new_port', 
                    encoded[action].unsqueeze(0), 
                    action,
                    dist.log_prob(action), 
                    value
                )'''
                state, value = self._get_state_value(encoded, action, self.new_port_critic)
                loacl_act, local_lp = self._localise_action()
                self._store_transition(
                    'new_port',
                    state,
                    loacl_act,
                    local_lp,
                    value
                )

                selected_idx = action.item()
                result['valid'] = True
                result['selection'] = node_ids[selected_idx]
                result['validation']['selected_utilization'] = features[selected_idx][0]
            
            return result
            
        except Exception as e:
            LOG.error(f"Error selecting new port: {e}")
            return {
                'valid': False,
                'selection': None,
                'validation': {'error': f'Exception in new port selection: {e}'},
                'encoded_state': None
            }
    
    def _extract_edges(self, graph_viz, edge_type, node_list):
        """Extract edge indices for specific edge type"""
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        edges = []
        
        for edge in graph_viz['edges']:
            edge_type_match = edge.get('edge_type')  or  edge.get('type')
            if edge_type_match == edge_type:
                src = edge['from']
                dst = edge['to']
                if src in node_to_idx and dst in node_to_idx:
                    edges.append([node_to_idx[src], node_to_idx[dst]])
        
        if not edges:
            # Self-loop if no edges
            edges = [[0, 0]]
        
        return torch.LongTensor(edges).t().to(self.device)
    
    # 3.  _store_transition  ▸  keep the *old* log-prob separately
    #    (replace the current definition)
    def _as_row(self, t: torch.Tensor) -> torch.Tensor:
        """    Return a 1-D tensor with exactly one element (shape == (1,)).
        Accepts scalar, (1,), or (1,1) and flattens when necessary."""
        '''if t.dim() == 0:                         # scalar  → (1,)
            return t.view(1)
        if t.dim() == 1 and t.numel() == 1:      # already OK
            return t
        raise ValueError(f"Bad tensor shape {tuple(t.shape)} – expected scalar or (1,)")'''
        if t.numel() != 1:
            raise ValueError(f"Bad tensor shape {tuple(t.shape)};"
                             " expected 1 element")
        return t.view(1)  # works for 0-D, (1,), or (1,1)

    def _store_transition(
        self,
        level: str,
        state: torch.Tensor,           # shape (1, H) — after _prepare_state
        action: torch.Tensor,          # ALWAYS 0 after localisation
        log_prob_old: torch.Tensor,    # ← untouched original log-prob
        value: torch.Tensor            # critic(s) output, shape (1,)
    ):
        """
        Save one on-policy sample.  We keep *log_prob_old* because PPO
        needs the probability *under the behaviour policy* that produced
        the action.
        """
        
        state = self._prepare_state(state, None)  # <<–– NEW LINE
        action = self._as_row(action.cpu())  # ensure shape (1,)
        log_prob_old = self._as_row(log_prob_old.cpu())  # ensure shape (1,)
        value = self._as_row(value.cpu())

        self.memory[level].append({
            "state"      : state.cpu(),
            "action"     : action,          # = tensor([0])
            "log_old"    : log_prob_old,    # any real value
            "value"      : value
        })


    def _store_transition1(self, level, state, action, log_prob, value):
        """Store transition in memory"""
        state = self._prepare_state(state, action)  # <<–– NEW LINE
        self.memory.append({
            'level': level,
            'state': state.cpu().detach(),
            'action': action.cpu().detach(),
            'log_prob': log_prob.cpu().detach(),
            'value': value.cpu().detach()
        })

    def _select_switch_with_validation(self, head1_data, graph_viz, switch_data=None):
        """
        Select switch with detailed validation tracking
        
        Returns:
            dict: {
                'valid': bool,
                'selection': switch_id or None,
                'validation': {...},
                'encoded_state': tensor or None
            }
        """
        try:
            result = {
                'valid': False,
                'selection': None,
                'validation': {},
                'encoded_state': None
            }
            
            # Get switch nodes
            switch_nodes = [
                n for n in graph_viz['nodes'] 
                if n.get('type') == 'switch' or n.get('head') == 1
            ]
            
            if not switch_nodes:
                result['validation']['error'] = 'No switch nodes found'
                return result
            
            # Prepare features and check flow counts
            features = []
            node_ids = []
            valid_switches = []
            
            

            for node in switch_nodes:
                mlu = 0.0
                flow_cnt = 0
                #LOG.info(f"[H_G_PPO-logging]: Processing switch node {node['id']}")
                #LOG.info(f"[H_G_PPO-logging]: node data: {node}")    
                # Try to get data from head1_data
                if head1_data and 'nodes' in head1_data:
                    for head_node in head1_data['nodes']:
                        #LOG.info(f"[H_G_PPO-logging]: Checking head_node {head_node['id']} against switch node {node['id']}")
                        if head_node.get('id') == node.get('id'):
                            mlu = head_node['data'].get('mlu', 0.0)
                            flow_cnt = head_node['data'].get('flow_count', 0)
                            #LOG.info(f"[H_G_PPO-logging]: head_node {head_node} for switch {node['id']}")
                            #LOG.info(f"[H_G_PPO-logging]: head1_data {head1_data} for switch {node['id']}")
                            LOG.info(f"[H_G_PPO-logging]: Found MLU {mlu} and flow count {flow_cnt} for switch {node['id']}")
                            break
                
                features.append([mlu, flow_cnt])
                node_ids.append(node['id'])
                
                # Track switches with flows
                if flow_cnt > 0:
                    valid_switches.append(True)
                else:
                    valid_switches.append(False)
            
            # Check if any switch has flows
            if not any(valid_switches):
                result['validation']['error'] = 'No switches with flows available'
                result['validation']['flow_counts'] = [f[1] for f in features]
                return result
            
            # Convert to tensor and encode
            x = torch.FloatTensor(features).to(self.device)
            edge_index = self._extract_edges(graph_viz, 'switch_link', node_ids)
            
            with torch.no_grad():
                #encoded = self.switch_encoder(x, edge_index)
                encoded = self._encode_eval(self.switch_encoder, x, edge_index)
                result['encoded_state'] = encoded
                
                logits = self.switch_actor(encoded).squeeze(-1)
                
                # Apply masking for switches with no flows
                flow_tensor = torch.tensor([f[1] for f in features], device=self.device)
                invalid_mask = (flow_tensor == 0)
                
                if self.config.get('mask_invalid_actions', True):
                    logits = self._apply_mask(logits, invalid_mask)
                    if logits is None:  # All switches invalid
                        result['validation']['error'] = 'All switches masked due to no flows'
                        return result
                
                # Add exploration noise
                if self.exploration_rate > self.config['min_exploration']:
                    logits = logits + torch.randn_like(logits) * self.exploration_rate
                
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                action = dist.sample()
                
                
                #value = self.switch_critic(encoded)
                # Store transition
                #self._store_transition('switch', encoded, action, dist.log_prob(action), value)
                #state, value = self._get_state_value(encoded, action, self.switch_critic)
                #self._store_transition('switch', state, action, dist.log_prob(action), value)
                #local_act, local_lp = self._localise_action()
                #self._store_transition(
                #    'switch',
                #    state,
                #    local_act,
                #    local_lp,
                #    value
                #)
                state, value = self._get_state_value(encoded, action, self.switch_critic)
                local_act, _ = self._localise_action()
                self._store_transition(
                    'switch',
                    state,
                    local_act,
                    dist.log_prob(action),
                    value
                )
                
                selected_idx = action.item()
                result['valid'] = True
                result['selection'] = node_ids[selected_idx]
                result['validation']['selected_flow_count'] = features[selected_idx][1]
                result['validation']['total_switches'] = len(switch_nodes)
                result['validation']['valid_switches'] = sum(valid_switches)
            
            return result
            
        except Exception as e:
            return {
                'valid': False,
                'selection': None,
                'validation': {'error': f'Exception in switch selection: {e}'},
                'encoded_state': None
            }

    def _store_partial_failure(self, failed_level, encoded_state=None):
        """
        Store partial failure for learning
        
        Args:
            failed_level: Level where failure occurred ('switch', 'port', 'flow', 'new_port')
            encoded_state: Encoded state if available
        """
        if not self.config.get('enable_partial_rewards', False):
            return
            
        if encoded_state is None:
            LOG.warning(f"[H_G_PPO-logging]: No encoded state provided for partial failure storage {failed_level}")
            return
        # Calculate partial penalty based on failure level
        penalties = self.config.get('partial_penalties', {})

        penalty_map = {
            'switch': penalties.get('switch_no_flows', -0.9),
            'port': penalties.get('port_no_flows', -0.8),
            'flow': penalties.get('flow_selection_failed', -0.3),  # usually not used
            'new_port': penalties.get('new_port_selection_failed', -0.2)  # usually not used
        }

        penalty = penalty_map.get(failed_level, -0.1)  # Default penalty

        dummy_action = torch.tensor(0, device=self.device)  # Dummy action for storage
        dummy_log_prob = torch.tensor(0.0, device=self.device)  # Dummy log prob

        critic = getattr(self, f"{failed_level}_critic")

        with torch.no_grad():
            # Get dummy value for the failed level
            value = critic(encoded_state).squeeze()

        state = self._prepare_state(encoded_state)
        self.memory[failed_level].append({
            'state': state.cpu(),
            'action': torch.zeros(1,dtype=torch.long),
            'log_prob': torch.zeros(1),
            'value': value.cpu(),
            'reward': penalty, # Store penalty as reward
        })

        
        
        if self.logging:
            print(f"[H_G_PPO-logging]: Stored partial failure at {failed_level} with penalty {penalty}")
    
    def calculate_reward2(self, prev_mlu, current_mlu, action):
        """Calculate reward including invalid levels tracking"""
        # Determine invalid levels from action validation
        invalid_levels = []
        
        if action.get('validation'):
            for level, validation_info in action['validation'].items():
                if validation_info.get('error'):
                    invalid_levels.append(level)
        
        # Calculate reward with invalid levels
        base_reward = calculate_reward(
            prev_mlu, current_mlu, action, self.config, invalid_levels
        )
        
        

        # Add partial rewards if enabled
        if (self.config.get('enable_partial_rewards', False) and 
            action.get('partial_reward_eligible', False)):
            
            partial_reward = self._calculate_partial_reward(action)
            total_reward = base_reward + partial_reward
            
            if self.logging:
                print(f"[H_G_PPO-logging]: Base reward: {base_reward}, "
                    f"Partial reward: {partial_reward}, Total: {total_reward}")
            
            return total_reward
        
        return base_reward
    
    def calculate_reward1(self, prev_mlu, current_mlu, action):
        """Calculate reward including partial rewards"""
        base_reward = calculate_reward(prev_mlu, current_mlu, action, self.config)
        
        # Add partial rewards if enabled
        if (self.config.get('enable_partial_rewards', False) and 
            action.get('partial_reward_eligible', False)):
            
            partial_reward = self._calculate_partial_reward(action)
            total_reward = base_reward + partial_reward
            
            if self.logging:
                print(f"[H_G_PPO-logging]: Base reward: {base_reward}, Partial reward: {partial_reward}, Total: {total_reward}")
            
            return total_reward
        
        return base_reward

    def _calculate_partial_reward(self, action):
        """Calculate partial reward based on action progress"""
        if not action.get('partial_progress'):
            return 0.0
        
        weights = self.config.get('partial_reward_weights', {
            'switch': 0.6, 'port': 0.4, 'flow': 0.3, 'new_port': 0.2
        })
        
        partial_reward = 0.0
        for level in action['partial_progress']:
            partial_reward += weights.get(level, 0.1)
        
        # Apply penalty if action failed before completion
        if action.get('failed_at_level') and not action.get('valid', False):
            penalty_factor = 0.5  # Reduce reward by 50% for incomplete actions
            partial_reward *= penalty_factor
        
        return partial_reward



    def calculate_reward1(self, prev_mlu, current_mlu, action):
        """Calculate reward using utility function"""
        return calculate_reward(
            prev_mlu, current_mlu, action, self.config
        )
    
    def update(self, reward, new_state):
        """Collect reward, adjust exploration, and trigger PPO."""
        self.episode_rewards.append(reward)

        # attach reward to LAST transition of each level
        for lvl, dq in self.memory.items():
            if dq and 'reward' not in dq[-1]:
                dq[-1]['reward'] = torch.tensor(reward, device=self.device)

        # exploration-rate decay
        self.exploration_rate = max(
            self.exploration_rate * self.config['exploration_decay'],
            self.config['min_exploration']
        )

        # tensorboard: ratio of partial failures across ALL transitions
        if hasattr(self, 'tb_writer'):
            flat_buf = [tr for dq in self.memory.values() for tr in dq]
            if flat_buf:
                ratio = sum(tr.get('partial_reward_eligible', False) for tr in flat_buf) / len(flat_buf)
                self.tb_writer.add_scalar('Training/partial_failures_ratio', ratio, self.training_step)

        # log & update
        LOG.info("[H_G_PPO-logging]: Calling _ppo_update()…")
        self._ppo_update()

    def update1(self, reward, new_state):
        """Update agent with PPO"""
        # Add reward to episode
        self.episode_rewards.append(reward)
        for lvl in self.memory:
            if len(self.memory[lvl]) and 'reward' not in self.memory[lvl][-1]:
                # Store reward in last transition if not already present
                self.memory[lvl][-1]['reward'] = reward
                #self.memory[lvl][-1]['done'] = False  # Mark as not done


        # Update exploration rate
        self.exploration_rate *= self.config['exploration_decay']
        self.exploration_rate = max(
            self.exploration_rate, 
            self.config['min_exploration']
        )

        # Log partial failures ratio
        if hasattr(self, 'tb_writer'):
            partial_count = sum(1 for t in self.memory if t.get('partial_reward_eligible', False))
            total_count = len(self.memory)
            if total_count > 0:
                self.tb_writer.add_scalar('Training/partial_failures_ratio',
                                          partial_count / total_count, 
                                          self.training_step)
        # inside update()
        for t in self.memory:
            if 'reward' not in t:
                t['reward'] = torch.tensor(reward)        # same reward for all steps


        # Perform PPO update if enough samples
        LOG.info("[H_G_PPO-logging]: Checking before calling _ppo_update()...")
        LOG.info(f"[H_G_PPO-logging]: Memory size: {len(self.memory)}")
        if len(self.memory) >= self.config['min_memory_size']:
            self._ppo_update()
    
    # hierarchical_gnn_ppo.py  ── inside class HierarchicalGNNPPOAgent
    def _ppo_update(self) -> None:
        """Run one PPO step *independently* for each hierarchy level."""
        LOG.info("[H_G_PPO-logging]: Performing PPO update for all levels...")


        for lvl in ['switch', 'port', 'flow', 'new_port']:
            buf = self.memory[lvl]
            LOG.info(f"[H_G_PPO-logging]: Level {lvl} has {len(buf)} transitions.")
            if len(buf) < self.min_mem:
                LOG.warning(f"[H_G_PPO-logging]: Not enough samples for level {lvl} ")
                continue                      # not enough samples yet
            keys = ('action', 'log_old', 'value')
            for k in keys:
                shapes = {tuple(t[k].shape) for t in buf}
                if len(shapes) > 1:
                    LOG.error(f"[H_G_PPO-logging]: Inconsistent {k} shapes for level {lvl}: {shapes}")
                    LOG.info(f"[H_G_PPO-logging]: Inconsistent {k} shapes for level {lvl}: {shapes}")
                    buf.clear()
            dims = {tuple(t['state'].shape) for t in buf}
            if len(dims) > 1:
                LOG.error(f"[H_G_PPO-logging]: Inconsistent state shapes for level {lvl}: {dims}")
                buf.clear()             # clear buffer
                LOG.warning(f"[H_G_PPO-logging]: Cleared buffer for level {lvl} due to inconsistent shapes.")
                continue                          # inconsistent shapes
            # pack batch ---------------------------------------------------
            states  = torch.cat([t['state']   for t in buf]).to(self.device)
            actions = torch.cat([t['action']  for t in buf]).to(self.device)
            log_old = torch.cat([t['log_old'] for t in buf]).to(self.device)
            rets    = torch.cat([t['reward']  for t in buf]).to(self.device)
            values  = torch.cat([t['value']   for t in buf]).to(self.device)

            # advantage ----------------------------------------------------
            adv = (rets - values)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # single-category dist ----------------------------------------
            dist    = torch.distributions.Categorical(
                        probs=torch.ones_like(actions).unsqueeze(1))
            ratio   = torch.exp(-log_old)     # log_new is zero
            surr1   = ratio * adv
            surr2   = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * adv

            actor_loss  = -torch.min(surr1, surr2).mean()
            critic_loss =  0.5 * (rets - values).pow(2).mean()

            loss = actor_loss + critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # logging ------------------------------------------------------
            self.step += 1
            LOG.info(f"[H_G_PPO-logging]: [{lvl}] step {self.step:4d} "
                    f"R {rets.mean():.3f}  AL {actor_loss:.3f}  CL {critic_loss:.3f}")

            buf.clear()                       # stay on-policy

    # ------------------------------------------------------------------
    # PPO update using the small, uniform transitions we now store
    # (state ▸ [1,H], action ▸ 0, log_old ▸ original log-prob).
    # ------------------------------------------------------------------
    def _ppo_update2(self) -> None:
        """
        Run *one* PPO optimiser step on *all* currently buffered transitions,
        then clear the buffer.  Works for every hierarchy level because all
        transitions have identical shapes.
        """
        if not self.memory:                      # empty buffer → nothing to do
            return

        shapes = [t["state"].shape for t in self.memory]
        assert all(s[0] == 1 for s in shapes), f"Inconsistent state shapes: {shapes}"

        # ── 1.  concatenate everything into flat tensors  ───────────────
        # Each key below was written by _store_transition().
        states   = torch.cat([t["state"]   for t in self.memory]).to(self.device)    # (B, H)
        actions  = torch.cat([t["action"]  for t in self.memory]).to(self.device)    # (B,) (all 0)
        log_old  = torch.cat([t["log_old"] for t in self.memory]).to(self.device)    # (B,)
        returns  = torch.cat([t["reward"]  for t in self.memory]).to(self.device)    # (B,)
        values   = torch.cat([t["value"]   for t in self.memory]).to(self.device)    # (B,)

        # ── 2.  advantage = (return − value) normalised  ────────────────
        adv = returns - values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ── 3.  build the *current* distribution for the collapsed state ─
        # With only one admissible action its prob vector is [1.0], so
        #   log_new = 0   and   entropy  = 0.
        dist     = torch.distributions.Categorical(
                       probs=torch.ones_like(actions, dtype=torch.float32).unsqueeze(1)
                   )
        log_new  = dist.log_prob(actions)          # zeros
        ratio    = torch.exp(log_new - log_old)     # = exp(-log_old)

        # ── 4.  PPO objective  (clip surrogate + value loss)  ───────────
        surr1       = ratio * adv
        surr2       = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * adv
        actor_loss  = -torch.min(surr1, surr2).mean()
        critic_loss =  0.5 * (returns - values).pow(2).mean()
        loss        = actor_loss + critic_loss      # (entropy term would be 0)

        # ── 5.  optimise  ───────────────────────────────────────────────
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ── 6.  book-keeping & cleanup  ─────────────────────────────────
        self.step += 1
        self.logger.info(
            f"Step {self.step} – AvgR {returns.mean():.4f}  "
            f"ALoss {actor_loss:.4f}  CLoss {critic_loss:.4f}"
        )
        self.memory.clear()        # ⇦ keep buffer on-policy for next cycle

    def _ppo_update1(self):
        """Perform PPO update with GAE, multiple epochs, and metrics tracking"""
        LOG.info("[H_G_PPO-logging]: Performing PPO update...")
        LOG.info(f"[H_G_PPO-logging]: Memory size: {len(self.memory)}")
        if len(self.memory) < self.config['batch_size']:
            LOG.warning("[H_G_PPO-logging]: Not enough samples for PPO update. "
                        "Skipping this update step.")
            return
        
        # Initialize metrics tracking
        actor_losses = {'switch': [], 'port': [], 'flow': [], 'new_port': []}
        critic_losses = {'switch': [], 'port': [], 'flow': [], 'new_port': []}
        entropy_bonuses = {'switch': [], 'port': [], 'flow': [], 'new_port': []}
        
        # Group transitions by level
        level_transitions = {'switch': [], 'port': [], 'flow': [], 'new_port': []}
        for transition in self.memory:
            level = transition['level']
            LOG.info(f"[H_G_PPO-logging]: Processing transition for level: {level}")
            if level in level_transitions:
                level_transitions[level].append(transition)
        LOG.info(f"[H_G_PPO-logging]: Level transitions: {level_transitions}")
        # PPO update for each level
        for level, transitions in level_transitions.items():
            if not transitions:
                continue
                
            # Get actor and critic for this level
            actor = getattr(self, f'{level}_actor')
            critic = getattr(self, f'{level}_critic')
            old_critic = getattr(self, f'{level}_critic')
            
            # Convert transitions to tensors
            states = torch.stack([t['state'] for t in transitions]).to(self.device)
            actions = torch.stack([t['action'] for t in transitions]).to(self.device)
            old_log_probs = torch.stack([t['log_prob'] for t in transitions]).to(self.device)
            old_values = torch.stack([t['value'] for t in transitions]).to(self.device)
            
            # Calculate advantages using GAE
            advantages = self._calculate_gae(transitions, old_values)
            returns = advantages + old_values.squeeze()
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO epochs
            for epoch in range(self.config['n_epochs']):
                LOG.info(f"[H_G_PPO-logging]: Starting epoch {epoch + 1} for level {level}")
                # Forward pass
                logits = actor(states)
                values = critic(states).squeeze()
                
                # Calculate probabilities and log probs
                if len(logits.shape) > 1:
                    logits = logits.squeeze(-1)
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                # Calculate ratio for PPO
                ratio = torch.exp(log_probs - old_log_probs)
                
                # Calculate surrogate losses
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 
                                1 - self.config['clip_epsilon'], 
                                1 + self.config['clip_epsilon']) * advantages
                
                # Actor loss (negative because we want to maximize)
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_clipped = old_values.squeeze() + torch.clamp(
                    values - old_values.squeeze(),
                    -self.config['clip_epsilon'],
                    self.config['clip_epsilon']
                )
                value_losses1 = (values - returns) ** 2
                value_losses2 = (value_clipped - returns) ** 2
                critic_loss = 0.5 * torch.max(value_losses1, value_losses2).mean()
                
                # Total loss
                loss = (actor_loss + 
                    self.config['value_loss_coef'] * critic_loss - 
                    self.config['entropy_coef'] * entropy)
                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()),
                    self.config['max_grad_norm']
                )
                
                self.optimizer.step()
                
                # Track metrics
                actor_losses[level].append(actor_loss.item())
                critic_losses[level].append(critic_loss.item())
                entropy_bonuses[level].append(entropy.item())
        
        # Store training metrics
        self._store_training_metrics(actor_losses, critic_losses, entropy_bonuses)
        
        # Clear memory after update
        self.memory.clear()
        
        LOG.debug(f"PPO update completed. Actor losses: {{k: np.mean(v) if v else 0 for k, v in actor_losses.items()}}")

    def _calculate_gae(self, transitions, values):
        """Calculate Generalized Advantage Estimation (GAE)"""
        # This is a simplified version - you'll need rewards from transitions
        # For now, using a placeholder
        advantages = []
        gae = 0
        
        # Note: In full implementation, you need to store rewards in transitions
        # and calculate proper TD errors
        for i in reversed(range(len(transitions))):
            # Get reward from transition (including penalties)
            reward = transitions[i].get('reward', 0)

            if i == len(transitions) - 1:
                # Last transition, next value is 0 (terminal state)
                next_value = 0  # Terminal state
                done = True  # Assuming last transition is terminal
            else:
                next_value = values[i + 1]
                # check if next transition is from a different episode
                done = transitions[i + 1].get('done', False)
            
            td_error = reward + (1 - done) * self.config['gamma'] * next_value - values[i]

            # Calculate GAE
            gae = td_error + (1 - done) * self.config['gamma'] * self.config['gae_lambda'] * gae
            advantages.insert(0, gae)

            #td_error = reward + self.config['gamma'] * next_value - values[i]
            #gae = td_error + self.config['gamma'] * self.config['gae_lambda'] * gae
            #advantages.insert(0, gae)
        
        return torch.tensor(advantages, device=self.device)

    def _store_training_metrics(self, actor_losses, critic_losses, entropy_bonuses):
        """Store training metrics to TensorBoard and CSV"""
        if not hasattr(self, 'training_metrics'):
            self.training_metrics = {
                'actor_losses': [],
                'critic_losses': [],
                'entropy_bonuses': [],
                'timestamps': []
            }
        
        # Calculate averages for each level
        metrics_dict = {}
        for level in ['switch', 'port', 'flow', 'new_port']:
            # Actor losses
            actor_loss = np.mean(actor_losses[level]) if actor_losses[level] else 0
            metrics_dict[f'actor_loss_{level}'] = actor_loss
            self.tb_writer.add_scalar(f'Loss/actor_{level}', actor_loss, self.training_step)
            
            # Critic losses
            critic_loss = np.mean(critic_losses[level]) if critic_losses[level] else 0
            metrics_dict[f'critic_loss_{level}'] = critic_loss
            self.tb_writer.add_scalar(f'Loss/critic_{level}', critic_loss, self.training_step)
            
            # Entropy
            entropy = np.mean(entropy_bonuses[level]) if entropy_bonuses[level] else 0
            metrics_dict[f'entropy_{level}'] = entropy
            self.tb_writer.add_scalar(f'Entropy/{level}', entropy, self.training_step)
        
        # Overall metrics
        avg_actor_loss = np.mean([metrics_dict[f'actor_loss_{l}'] for l in ['switch', 'port', 'flow', 'new_port']])
        avg_critic_loss = np.mean([metrics_dict[f'critic_loss_{l}'] for l in ['switch', 'port', 'flow', 'new_port']])
        avg_entropy = np.mean([metrics_dict[f'entropy_{l}'] for l in ['switch', 'port', 'flow', 'new_port']])
        
        self.tb_writer.add_scalar('Loss/actor_avg', avg_actor_loss, self.training_step)
        self.tb_writer.add_scalar('Loss/critic_avg', avg_critic_loss, self.training_step)
        self.tb_writer.add_scalar('Entropy/avg', avg_entropy, self.training_step)
        
        # Calculate average reward over last 100 episodes
        avg_reward_100 = np.mean(self.reward_history) if self.reward_history else 0
        self.tb_writer.add_scalar('Reward/avg_100', avg_reward_100, self.training_step)
        
        # Log exploration rate
        self.tb_writer.add_scalar('Exploration/rate', self.exploration_rate, self.training_step)
        
        # Write to CSV
        self._write_metrics_to_csv(metrics_dict, avg_reward_100)
        
        # Flush TensorBoard to ensure real-time updates
        self.tb_writer.flush()
        
        # Store in memory
        self.training_metrics['actor_losses'].append(avg_actor_loss)
        self.training_metrics['critic_losses'].append(avg_critic_loss)
        self.training_metrics['entropy_bonuses'].append(avg_entropy)
        self.training_metrics['timestamps'].append(time.time())
        
        # Log to console periodically
        if self.training_step % 10 == 0:
            LOG.info(f"Step {self.training_step} - Avg Reward: {avg_reward_100:.4f}, "
                    f"Actor loss: {avg_actor_loss:.4f}, "
                    f"Critic loss: {avg_critic_loss:.4f}, "
                    f"Entropy: {avg_entropy:.4f}")
    
    def save_model(self, path):
        """Save model checkpoint"""
        checkpoint = {
            'switch_encoder': self.switch_encoder.state_dict(),
            'port_encoder': self.port_encoder.state_dict(),
            'flow_encoder': self.flow_encoder.state_dict(),
            'new_port_encoder': self.new_port_encoder.state_dict(),
            'switch_actor': self.switch_actor.state_dict(),
            'port_actor': self.port_actor.state_dict(),
            'flow_actor': self.flow_actor.state_dict(),
            'new_port_actor': self.new_port_actor.state_dict(),
            'switch_critic': self.switch_critic.state_dict(),
            'port_critic': self.port_critic.state_dict(),
            'flow_critic': self.flow_critic.state_dict(),
            'new_port_critic': self.new_port_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'exploration_rate': self.exploration_rate,
            'config': self.config
        }
        torch.save(checkpoint, path)
        LOG.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.switch_encoder.load_state_dict(checkpoint['switch_encoder'])
        self.port_encoder.load_state_dict(checkpoint['port_encoder'])
        self.flow_encoder.load_state_dict(checkpoint['flow_encoder'])
        self.new_port_encoder.load_state_dict(checkpoint['new_port_encoder'])
        
        self.switch_actor.load_state_dict(checkpoint['switch_actor'])
        self.port_actor.load_state_dict(checkpoint['port_actor'])
        self.flow_actor.load_state_dict(checkpoint['flow_actor'])
        self.new_port_actor.load_state_dict(checkpoint['new_port_actor'])
        
        self.switch_critic.load_state_dict(checkpoint['switch_critic'])
        self.port_critic.load_state_dict(checkpoint['port_critic'])
        self.flow_critic.load_state_dict(checkpoint['flow_critic'])
        self.new_port_critic.load_state_dict(checkpoint['new_port_critic'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.exploration_rate = checkpoint.get('exploration_rate', 0.1)
        
        LOG.info(f"Model loaded from {path}")


    def _init_csv_logging(self):
        """Initialize CSV file with headers"""
        headers = [
            'timestamp', 'training_step', 'episode_reward', 'avg_reward_100',
            'actor_loss_switch', 'actor_loss_port', 'actor_loss_flow', 'actor_loss_new_port',
            'critic_loss_switch', 'critic_loss_port', 'critic_loss_flow', 'critic_loss_new_port',
            'entropy_switch', 'entropy_port', 'entropy_flow', 'entropy_new_port',
            'exploration_rate', 'mlu_improvement'
        ]
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)


    def _write_metrics_to_csv(self, metrics_dict, avg_reward_100):
        """Write metrics to CSV file"""
        row = [
            datetime.now().isoformat(),
            self.training_step,
            self.reward_history[-1] if self.reward_history else 0,  # Latest reward
            avg_reward_100,
            metrics_dict.get('actor_loss_switch', 0),
            metrics_dict.get('actor_loss_port', 0),
            metrics_dict.get('actor_loss_flow', 0),
            metrics_dict.get('actor_loss_new_port', 0),
            metrics_dict.get('critic_loss_switch', 0),
            metrics_dict.get('critic_loss_port', 0),
            metrics_dict.get('critic_loss_flow', 0),
            metrics_dict.get('critic_loss_new_port', 0),
            metrics_dict.get('entropy_switch', 0),
            metrics_dict.get('entropy_port', 0),
            metrics_dict.get('entropy_flow', 0),
            metrics_dict.get('entropy_new_port', 0),
            self.exploration_rate,
            getattr(self, 'last_mlu_improvement', 0)  # Use stored MLU improvement
        ]
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'tb_writer'):
            self.tb_writer.close()
            LOG.info(f"TensorBoard writer closed. Metrics saved to {self.metrics_dir}")


    def calculate_reward(self, prev_mlu, current_mlu, action):
        """
        Calculate reward including partial rewards and logging
        
        Args:
            prev_mlu: Previous maximum link utilization
            current_mlu: Current maximum link utilization  
            action: Action taken with validation details
            
        Returns:
            float: Total reward (base + partial if enabled)
        """
        # Determine invalid levels from action validation
        invalid_levels = []
        if action.get('validation'):
            for level, validation_info in action['validation'].items():
                if validation_info.get('error'):
                    invalid_levels.append(level)
        
        # Calculate base reward using utility function
        base_reward = calculate_reward(
            prev_mlu, current_mlu, action, self.config, invalid_levels
        )
        
        # Calculate partial reward if enabled
        partial_reward = 0.0
        if (self.config.get('enable_partial_rewards', False) and 
            action.get('partial_reward_eligible', False)):
            partial_reward = self._calculate_partial_reward(action)
        
        # Total reward
        total_reward = base_reward + partial_reward
        
        # MLU improvement
        mlu_improvement = prev_mlu - current_mlu
        
        # ============ LOGGING SECTION ============
        # Log to TensorBoard immediately
        if hasattr(self, 'tb_writer'):
            # Reward metrics
            self.tb_writer.add_scalar('Reward/immediate', total_reward, self.training_step)
            self.tb_writer.add_scalar('Reward/base', base_reward, self.training_step)
            self.tb_writer.add_scalar('Reward/partial', partial_reward, self.training_step)
            
            # MLU metrics
            self.tb_writer.add_scalar('MLU/previous', prev_mlu, self.training_step)
            self.tb_writer.add_scalar('MLU/current', current_mlu, self.training_step)
            self.tb_writer.add_scalar('MLU/improvement', mlu_improvement, self.training_step)
            
            # Action validity metrics
            action_valid = action.get('valid', False)
            self.tb_writer.add_scalar('Action/success_rate', 
                                    1.0 if action_valid else 0.0, 
                                    self.training_step)
            
            # Log failure details if action failed
            if not action_valid:
                failed_level = action.get('failed_at_level', 'unknown')
                self.tb_writer.add_text('Action/failure', 
                                        f"Failed at: {failed_level} | Step: {self.training_step}", 
                                        self.training_step)
                
                # Log validation errors
                if action.get('validation'):
                    for level, val_info in action['validation'].items():
                        if val_info.get('error'):
                            self.tb_writer.add_text(f'Validation/{level}_error', 
                                                    val_info['error'], 
                                                    self.training_step)
            
            # Log action progress
            progress_count = len(action.get('partial_progress', []))
            self.tb_writer.add_scalar('Action/progress_levels', progress_count, self.training_step)
            
            # Flush to ensure real-time updates
            self.tb_writer.flush()
        
        # Store in reward history
        if hasattr(self, 'reward_history'):
            self.reward_history.append(total_reward)
        
        # Log to console if enabled
        if self.logging:
            LOG.info(f"[H_G_PPO-logging] Step {self.training_step}: "
                    f"Reward={total_reward:.4f} (base={base_reward:.4f}, partial={partial_reward:.4f}), "
                    f"MLU: {prev_mlu:.4f} -> {current_mlu:.4f} (Δ={mlu_improvement:.4f}), "
                    f"Action valid: {action.get('valid', False)}")
        
        # Store MLU improvement for CSV logging
        self.last_mlu_improvement = mlu_improvement
        
        return total_reward
    
    # hierarchical_gnn_ppo.py
# ──────────────────────────────────────────────────────────────────────────────
# Add this helper just above _store_transition (or anywhere inside the class
# before _store_transition is defined).

    def _prepare_state(self, state: torch.Tensor, action: Optional[torch.Tensor] = None):
        """
        Ensure that the state we put in the replay buffer is a SINGLE embedding
        vector of shape (1, H).

        • If `state` is (N, H) and we know which row was chosen (`action` is
          not None), keep *only* that row → (1, H).
        • If `state` is (N, H) but we don’t know which row (e.g., partial
          failure), fall back to the mean over rows → (1, H).  This avoids
          exploding the dimensionality while still giving the critic *some*
          information.
        • If `state` is already (H,) or (1, H) we reshape to (1, H).
        """
        if state.dim() == 2 and state.size(0) > 1:          # many nodes
            if action is not None:                          # we picked one
                state = state[action].unsqueeze(0)          # → (1, H)
            else:                                           # no index ⇒ mean
                state = state.mean(dim=0, keepdim=True)     # → (1, H)
        elif state.dim() == 1:                              # (H,) ⇒ (1, H)
            state = state.unsqueeze(0)
        # state is now guaranteed (1, H)
        return state
