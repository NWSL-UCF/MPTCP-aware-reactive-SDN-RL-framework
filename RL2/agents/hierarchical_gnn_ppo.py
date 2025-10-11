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
from typing import Optional, Tuple, Dict, List  # ← std-lib typing helpers

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
        self.opt = {
            'switch' : torch.optim.Adam(
                list(self.switch_encoder.parameters()) +
                list(self.switch_actor.parameters()) +
                list(self.switch_critic.parameters()),
                lr=config['lr']['switch']
            ),
            'port' : torch.optim.Adam(
                list(self.port_encoder.parameters()) +
                list(self.port_actor.parameters()) +
                list(self.port_critic.parameters()),
                lr=config['lr']['port']
            ),
            'flow' : torch.optim.Adam(
                list(self.flow_encoder.parameters()) +
                list(self.flow_actor.parameters()) +
                list(self.flow_critic.parameters()),
                lr=config['lr']['flow']
            ),
            'new_port' : torch.optim.Adam(
                list(self.new_port_encoder.parameters()) +
                list(self.new_port_actor.parameters()) +
                list(self.new_port_critic.parameters()),
                lr=config['lr']['new_port']
            ),
        }
        
        # Memory and tracking [per hierarchy level]
        self.memory = {lvl: deque(maxlen=config['memory_size']) for lvl in ['switch', 'port', 'flow', 'new_port']}
        
        self.reward_history = deque(maxlen=100)  # Overall reward history for averaging

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
        #self.reward_history = deque(maxlen=100)  # Keep last 100 rewards for averaging
        self.switch_reward_history = deque(maxlen=100)
        self.port_reward_history = deque(maxlen=100)
        self.flow_reward_history = deque(maxlen=100)
        self.new_port_reward_history = deque(maxlen=100)
        self.step = 0
        self.update_every = 4
        self.min_mem = config['min_memory_size']
        
        LOG.info(f"Hierarchical GNN-PPO Agent initialized on {self.device}")
        LOG.info(f"Metrics will be saved to: {self.metrics_dir}")
    
    ACTOR = property(lambda self: {
        'switch'  : self.switch_actor,
        'port'    : self.port_actor,
        'flow'    : self.flow_actor,
        'new_port': self.new_port_actor
    })

    CRITIC = property(lambda self: {
        'switch'  : self.switch_critic,
        'port'    : self.port_critic,
        'flow'    : self.flow_critic,
        'new_port': self.new_port_critic
    })

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
                'partial_reward_eligible': self.config.get('enable_partial_rewards', False),
                'is_no_op': False,  # Track if no action is needed
                'no_op_level': None  # Track no-op level if applicable
            }
            #LOG.info(f"[H_G_PPO-logging]: memory size: {len(self.memory)}")
            # Extract head data
            head_data = graph_data.get('head_data', {})
            pyg_data = graph_data.get('pyg_data', {})
            graph_viz = graph_data.get('graph_viz', {'nodes': [], 'edges': []})
            
            
            # Track total flows in the system
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
                # NEW: Validate selected switch has flows
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

                    return action  #  Return immediately, don't proceed to port selection
                if self.logging:
                    LOG.info(f"[H_G_PPO-logging]: Selected switch: {switch_result['selection']}")
                if switch_result.get('is_no_op', False):
                    action['is_no_op'] = True
                    action['no_op_level'] = 'switch'
                    action['valid'] = True  # No-op is still a valid action
                    action['validation']['switch'] = switch_result['validation']
                    
                    # Store the network MLU for the reward calculation
                    action['validation']['network_mlu'] = network_state.get('mlu',0.0)
                    if self.logging:
                        LOG.info(f"[H_G_PPO-logging]: No-op at switch level, no action needed")
                    return action
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


                if port_result.get('is_no_op', False):
                    action['is_no_op'] = True
                    action['no_op_level'] = 'port'
                    action['valid'] = True

                    action['validation']['pre_action_mlu'] = network_state.get('mlu', 0.0)
                    action['validation']['switch_mlus'] = network_state.get('switch_mlus', {})

                    if self.logging:
                        LOG.info(f"[H_G_PPO-logging]: No-op at port level, no action needed")

                    return action # ✅ Return immediately, no further action needed

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
                port_result['selection'], graph_viz, pyg_data.get(3), network_state=network_state
            )
            
            if flow_result['valid']:
                action['flow'] = flow_result['selection']
                action['partial_progress'].append('flow')
                action['validation']['flow'] = flow_result['validation']
                
                if self.logging:
                    print(f"[H_G_PPO-logging]: Selected flow: {flow_result['selection']}")
                if flow_result.get('is_no_op', False):
                    action['is_no_op'] = True
                    action['no_op_level'] = 'flow'
                    action['valid'] = True
                    action['validation']['flow_rate'] = flow_result['validation'].get('flow_rate', 0.0)

                    if self.logging:
                        LOG.info(f"[H_G_PPO-logging]: No-op at flow level, no action needed")
                    return action  # ✅ Return immediately, no further action needed

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

                if new_port_result.get('is_no_op', False):
                    action['is_no_op'] = True
                    action['no_op_level'] = 'new_port'
                    #action['validation']['new_port'] = new_port_result['validation']



                    
                    
                    if self.logging:
                        LOG.info(f"[H_G_PPO-logging]: No-op at new port level, no action needed")
                    return action
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
                'error': str(e),
                'is_no_op': False
            }

    def _select_port_with_validation1(self, switch_id, graph_viz, port_data=None):
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
                #LOG.info(f"[H_G_PPO-logging]: Processing port node: {node}")
                util = node['data'].get('utilization', 0.0)
                flow_count = node['data'].get('flow_count', 0)
                #LOG.info(f"[H_G_PPO-logging]: Port utilization: {util}, flow count: {flow_count}")
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
                #loacl_act, local_lp = self._localise_action()
                self._store_transition(
                    'port',
                    encoded,
                    action,
                    dist.log_prob(action),
                    value
                )

                selected_idx = action.item()
                result['valid'] = True
                result['selection'] = node_ids[selected_idx]
                result['validation']['selected_flow_count'] = features[selected_idx][1]
                result['validation']['utilization'] = features[selected_idx][0]
                result['validation']['total_ports'] = len(port_nodes)
                result['validation']['valid_ports'] = sum(valid_ports)
                #LOG.info(f"[H_G_PPO-logging]: Port selection result: {result}")

            return result
            
        except Exception as e:
            return {
                'valid': False,
                'selection': None,
                'validation': {'error': f'Exception in port selection: {e}'},
                'encoded_state': None
            }
    
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

    def _select_flow(self, port_id, graph_viz, flow_data=None, network_state=None):
        """Select flow from port"""
        try:
            result = {
                'valid': False,
                'selection': None,
                'validation': {},
                'encoded_state': None,
                'is_no_op': False  # Track if no-op is selected
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

            port_util = [n['data'].get('utilization', 0.0) for n in graph_viz['nodes'] if n['id'] == port_id]
            LOG.info(f"[H_G_PPO-logging]: Port utilization for {port_id}: {port_util}")
            # Add No-Op option first
            # No-Op features: [average rate, 0.0]
            flow_rates = [node['data'].get('rate', 0.0) for node in flow_nodes]
            avg_rate = np.mean(flow_rates) if flow_rates else 0.0
            max_rate = max(flow_rates) if flow_rates else 0.0
            features.append([avg_rate / max_rate if max_rate > 0 else 0, 0.0])  # No-Op has rate
            node_ids.append('no_op')  # Use a special ID for No-Op

            # Add actual flow features
            
            for node in flow_nodes:
                rate = node['data'].get('rate', 0.0)
                dst_dpid = float(node['data'].get('dst_dpid', 0))
                features.append([rate / max_rate if max_rate > 0 else 0, dst_dpid / 18.0])  # Normalize dst_dpid
                node_ids.append(node['id'])
            
            # Convert to tensor
            x = torch.FloatTensor(features).to(self.device)
            
            
            if flow_data is not None:
                # Shift edge_index to match node IDs
                edge_index = flow_data.edge_index.to(self.device) + 1

                # Add self-loop edges for No-Op
                no_op_self = torch.zeros((2, 1), dtype=torch.long, device=self.device)
                edge_index = torch.cat([edge_index, no_op_self], dim=1)
            else:
                # Create self-loop edges as fallback
                num_flows = len(features)
                edge_index = torch.LongTensor([[i, i] for i in range(num_flows)]).t().to(self.device)
            # create edge_index for No-Op node from port_id node


            # Simple encoding (can be enhanced with local GNN)
            with torch.no_grad():
                # For now, just use first layer of encoder
                #encoded = self.flow_encoder(x, edge_index)
                encoded = self._encode_eval(self.flow_encoder, x, edge_index)
                result['encoded_state'] = encoded
                
                # Get action probabilities
                logits = self.flow_actor(encoded).squeeze(-1)

                # Apply bias for No-Op
                if self.config.get('encourage_no_op', True):
                    #port_util = port_util
                    candidate_new_ports = []
                    candidate_new_ports_utils = []
            
                    for edge in graph_viz['edges']:
                        if (edge['from'] == flow_id and 
                            edge['edge_type'] == 'flow_to_new_port'):
                            port_id = edge['to']
                            # Find new port node
                            for node in graph_viz['nodes']:
                                if node['id'] == port_id and node['type'] == 'new_port':
                                    candidate_new_ports.append(node)
                                    node_util = node['data'].get('utilization', 0.0)
                                    candidate_new_ports_utils.append(node_util)
                                    break
                    if not candidate_new_ports:
                        LOG.warning("[H_G_PPO-logging]: No candidate new ports found for flow selection")
                        pass
                    network_mlu = network_state.get('mlu', 0.0)
                    #LOG.info(f"[H_G_PPO-logging]: Candidate new ports: {candidate_new_ports}")
                    #LOG.info(f"[H_G_PPO-logging]: Port utilization: {port_util}, "
                    #         f"Network MLU: {network_mlu}")
                    #LOG.info(f"[H_G_PPO-logging]: Candidate new ports: {candidate_new_ports_utils}")
                    bias, reason = self._should_encourage_flow_no_op(port_util, flow_rates, network_mlu, candidate_new_ports)
                    logits[0] += bias
                    LOG.info(f"[H_G_PPO-logging]: No-Op bias applied: {bias}, Reason: {reason}")
                    # print the top 5 logits for debugging
                    k = len(logits) if len(logits) < 5 else 5
                    top_logits = torch.topk(logits, k)
                    LOG.info(f"[H_G_PPO-logging]: Top {k} logits after bias: {top_logits.values.tolist()}")
                else:
                    candidate_new_ports_utils = []
            
                    for edge in graph_viz['edges']:
                        if (edge['from'] == flow_id and 
                            edge['edge_type'] == 'flow_to_new_port'):
                            port_id = edge['to']
                            # Find new port node
                            for node in graph_viz['nodes']:
                                if node['id'] == port_id and node['type'] == 'new_port':
                                    node_util = node['data'].get('utilization', 0.0)
                                    candidate_new_ports_utils.append(node_util)
                                    break
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
                #loacl_act, local_lp = self._localise_action()
                self._store_transition(
                    'flow',
                    encoded,
                    action,
                    dist.log_prob(action),
                    value
                )

                selected_idx = action.item()
                result['valid'] = True
                result['selection'] = node_ids[selected_idx]
                result['is_no_op'] = (selected_idx == 0)  # Check if No-Op selected
                if result['is_no_op']:
                    result['action_type'] = 'no_op'
                    result['validation']['selected_flow_rate'] = avg_rate
                    result['validation']['dst_dpid'] = 0.0
                    result['validation']['max_flow_rate'] = max_rate
                    result['validation']['candidate_new_ports_utils'] = candidate_new_ports_utils
                else:
                    result['validation']['selected_flow_rate'] = features[selected_idx][0]
                    result['validation']['dst_dpid'] = features[selected_idx][1]
                    result['validation']['max_flow_rate'] = max_rate

            return result
            
        except Exception as e:
            LOG.error(f"Error selecting flow: {e}")
            return {
                'valid': False,
                'selection': None,
                'validation': {'error': f'Exception in flow selection: {e}'},
                'encoded_state': None
            }
        
    def _should_encourage_flow_no_op(self, src_port_util, flow_rates, network_mlu, candidate_new_ports):
        bias = 0.0
        reasons = []
        
        # Calculate metrics
        src_util = src_port_util[0] if isinstance(src_port_util, list) else src_port_util
        avg_candidate_util = np.mean([p['data']['utilization'] for p in candidate_new_ports]) if candidate_new_ports else 1.0
        max_flow_rate = max(flow_rates) if flow_rates else 0
        is_elephant_present = any(rate >= 0.1 * setting.LINK_MAX_CAPACITY for rate in flow_rates)
        best_dest_util = min([p['data']['utilization'] for p in candidate_new_ports], default=1.0) if candidate_new_ports else 1.0
        delta_util = src_util - best_dest_util
        
        # Decision logic
        if not candidate_new_ports:
            bias += 5.0  # Strong bias - nowhere to go
            reasons.append("No destinations available")

        elif delta_util >= 0.2:
            bias -= 5.0 * delta_util  # Strong bias to stay put
        else:
            reasons.append(f"Source port utilization ({src_util:.2f}) is not significantly better than candidates ({avg_candidate_util:.2f})")
        # elif src_util > 0.8 and avg_candidate_util < src_util - 0.2:
        #     # Only migrate if destination is significantly better
        #     bias -= 2.0
        #     reasons.append(f"Source congested ({src_util:.2f}) and better destinations available ({avg_candidate_util:.2f})")
        
        # elif src_util < 0.5:
        #     # Source is not congested
        #     bias += 1.0
        #     reasons.append("Source port not congested")
        
        # # Flow-specific logic
        # if not is_elephant_present:
        #     bias += 1.5  # Don't bother migrating mice flows
        #     reasons.append("No elephant flows present")
        
        # # Network-wide considerations
        # if network_mlu > 0.8:
        #     # Network is stressed - be more careful about migrations
        #     bias += 0.5
        #     reasons.append("Network highly utilized")
        
        return np.clip(bias, -5.0, 5.0), " | ".join(reasons)

    def _should_encourage_flow_no_op1(self, 
                                     src_port_util: float, 
                                     flow_rates: List[float], 
                                     network_mlu: float,
                                     candidate_new_ports: List[Dict]) -> Tuple[float, str]:
        """
        Determine if No-Op should be encouraged at flow level
        
        Returns a bias value to add to No-Op logit
        """
        bias = 0.0
        reason = []
        LOG.info(f"[H_G_PPO-logging]: Evaluating No-Op encouragement: "
                 f"src_port_util={src_port_util}, flow_rates={flow_rates}, "
                 f"network_mlu={network_mlu}, candidate_new_ports={candidate_new_ports}")
        src_port_util = src_port_util[0] if isinstance(src_port_util, list) else src_port_util
        LOG.info(f"[H_G_PPO-logging]: Source port utilization: {src_port_util}")

        # 1. No destinations or all destinations full → stay put
        # If there are ports to reroute to, all ports congested
        #LOG.info(f"[H_G_PPO-logging]: number Candidate new ports: {len(candidate_new_ports)}")
        if len(candidate_new_ports) == 0:
            bias += 5.0; reason.append("No candidate new ports available")
        elif all(p['data']['utilization'] > 0.80 for p in candidate_new_ports):
            #LOG.info(f"[H_G_PPO-logging]: All candidate new ports are congested")
            bias += 2.0; reason.append("All candidate new ports are congested")
        elif all(p['data']['utilization'] < 0.30 for p in candidate_new_ports):
            bias -= 1.0; reason.append("All candidate new ports are underutilized")
        #LOG.info(f"[H_G_PPO-logging]: is congested: {all(p['data']['utilization'] > 0.80 for p in candidate_new_ports)}")
        #LOG.info(f"[H_G_PPO-logging]: Candidate new ports utilization: "
        #         f"{[p['data']['utilization'] for p in candidate_new_ports]}")
        #LOG.info(f"[H_G_PPO-logging]: No-Op bias after checking candidate ports: {bias}, Reason: {reason}")
        # 2. Source port heavily congested → *prefer* to migrate(reroute)
        # If port is highly congested, encourage No-Op
        if src_port_util > 0.80:  # Source port utilization > 80%
            bias -= 2.0; reason.append("Source port utilization is very high")
        
        # 3. Only tiny (mice) flows present → migration cost > benefit
        # If all flows are small (mice flows), maybe don't migrate
        max_flow_rate = max(flow_rates) if flow_rates else 0
        if max_flow_rate < 0.1 * setting.LINK_MAX_CAPACITY:  # All mice flows
            bias += 1.0; reason.append("All flows are small (mice flows)")
        #LOG.info(f"[H_G_PPO-logging]: Max flow rate: {max_flow_rate}, "
        #            f"Link max capacity: {setting.LINK_MAX_CAPACITY}")
        #LOG.info(f"[H_G_PPO-logging]: No-Op bias after checking flow rates: {bias}, Reason: {reason}")
        # 4. Fabric almost empty → migrating seldom helps
        # If network MLU is very low, slight bias toward No-Op
        if network_mlu < 0.3:
            bias += 0.5; reason.append("Network MLU is low")
        #LOG.info(f"[H_G_PPO-logging]: Network MLU: {network_mlu}, "
        #         f"No-Op bias after checking network MLU: {bias}, Reason: {reason}")
        # final clamp
        bias = min(max(bias, -3.0), 3.0)  # Limit bias range
        #LOG.info(f"[H_G_PPO-logging]: No-Op bias applied: {bias}, Reason: {reason}")
        return bias, "|".join(reason)

    def _select_new_port(self, flow_id, graph_viz, new_port_data=None):
        """Select new port for flow"""
        try:
            result = {
                'valid': False,
                'selection': None,
                'is_no_op': False,  # Track if No-Op is selected
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
            avg_utilization = 1 - (np.mean([node['data'].get('utilization', 0.0) for node in port_nodes]))
            # Add No-Op option first
            features.append([avg_utilization])  # No-Op has average utilization
            node_ids.append('no_op')  # Use a special ID for No-Op
            
            # Add actual port features

            #node_ids = [node['id'] for node in port_nodes]
            valid_ports = [True]
            for node in port_nodes:
                util = 1- node['data'].get('utilization', 0.0)
                features.append([util])
                node_ids.append(node['id'])
            

            LOG.info(f"[H_G_PPO-logging]: New port features: {features}, "
                     f"Node IDs: {node_ids}, Avg Utilization: {avg_utilization}")
            # Convert to tensor
            x = torch.FloatTensor(features).to(self.device)

            if new_port_data is not None:
                edge_index = new_port_data.edge_index.to(self.device) + 1  # Shift indices for No-Op

                # Add self-loop edges for No-Op
                no_op_self = torch.zeros((2, 1), dtype=torch.long, device=self.device)
                edge_index = torch.cat([edge_index, no_op_self], dim=1)
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
                #logits[0] -= 1.0  # Encourage No-Op by default
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
                #loacl_act, local_lp = self._localise_action()
                self._store_transition(
                    'new_port',
                    encoded,
                    action,
                    dist.log_prob(action),
                    value
                )

                selected_idx = action.item()
                result['is_no_op'] = (selected_idx == 0)  # Check if No-Op selected
                if result['is_no_op']:
                    result['selection'] = 'no_op'
                    result['validation']['selected_util'] = avg_utilization
                result['valid'] = True
                result['selection'] = node_ids[selected_idx]
                result['validation']['selected_utilization'] = features[selected_idx][0]
            #LOG.info(f"[H_G_PPO-logging]: New port node 0: {node_ids[0]}, "
            #         f"selected: {result['selection']}, "
            #         f"utilization: {result['validation']['selected_utilization']}")
            #LOG.info(f"*******[H_G_PPO-logging]: New port selection result: {result}")
            
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

    # ── hierarchical_gnn_ppo.py ──
    def _store_transition(
        self,
        level: str,
        state: torch.Tensor,          # [N, H] or [1, H]
        action_idx: torch.Tensor,     # scalar tensor
        log_prob_old: torch.Tensor,   # scalar tensor preferred
        value: torch.Tensor           # scalar or vector
    ) -> None:
        """
        Save one on-policy transition and guarantee that *action*,
        *log_prob_old* and *value* are single-element tensors so we never
        hit shape-mismatch errors.

        • If a vector is supplied for *value* or *log_prob_old* we keep the
          element that corresponds to the chosen *action_idx*.
        • Everything is finally stored on CPU (deque memory).
        """

        # ---- action ---------------------------------------------------------
        action_idx = action_idx.flatten()[:1].cpu()          # → shape (1,)

        # ---- log-prob -------------------------------------------------------
        if log_prob_old.numel() > 1:                         # vector → scalar
            log_prob_old = log_prob_old.flatten()[action_idx.item():action_idx.item()+1]
        else:
            log_prob_old = log_prob_old.view(1)
        log_prob_old = log_prob_old.cpu()                    # (1,)

        # ---- value ----------------------------------------------------------
        if value.numel() > 1:                                # vector → scalar
            value = value.flatten()[action_idx.item():action_idx.item()+1]
        else:
            value = value.view(1)
        value = value.cpu()                                  # (1,)

        # ---- store ----------------------------------------------------------
        self.memory[level].append({
            "state"   : state.cpu(),     # keep full embedding (variable-length)
            "action"  : action_idx,
            "log_old" : log_prob_old,
            "value"   : value
        })

    
    def _store_transition2(
            self, level: str,
            encoded: torch.Tensor,      # shape (N, H)  (no collapse!)
            action_idx: torch.Tensor,   # scalar index   (0-D or (1,))
            log_prob_old: torch.Tensor, # behaviour log-π(a)
            value: torch.Tensor         # critic output  (1,)
        ) -> None:
        """
        Save a *raw* graph embedding together with action and old log-prob.
        Reward will be attached later in `update()`.
        """
        self.memory[level].append({
            "state" : encoded.cpu(),               # variable length
            "action"  : action_idx.view(1).cpu(),    # (1,)
            "log_old" : log_prob_old.view(1).cpu(),  # (1,)
            "value"   : value.view(1).cpu(),         # (1,)
            # reward filled in update()
        })

    
    
    def _store_transition1(
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



    # In hierarchical_gnn_ppo.py, replace the existing selection methods with these:

    def _select_switch_with_validation(self, head1_data, graph_viz, switch_data=None):
        """
        Select switch with No-Op option and detailed validation tracking
        
        Returns:
            dict: {
                'valid': bool,
                'selection': switch_id or 'no_op',
                'validation': {...},
                'encoded_state': tensor or None,
                'is_no_op': bool
            }
        """
        try:
            result = {
                'valid': False,
                'selection': None,
                'validation': {},
                'encoded_state': None,
                'is_no_op': False
            }
            
            # Get switch nodes
            switch_nodes = [
                n for n in graph_viz['nodes'] 
                if n.get('type') == 'switch' or n.get('head') == 1
            ]
            
            if not switch_nodes:
                result['validation']['error'] = 'No switch nodes found'
                return result
            
            # Prepare features for actual switches
            features = []
            node_ids = []
            valid_switches = []
            max_cnt = 0
            # Add No-Op option (always valid)
            # No-Op gets special features: average MLU and total flow count
            avg_mlu = np.mean([f[0] for f in features]) if features else 0.0
            avg_flows = np.mean([f[1] for f in features]) if features else 1.0
            max_flows = max([f[1] for f in features], default=1.0)
            #total_flows = max(total_flows, 1)  # Avoid division by zero
            #LOG.info(f"[H_G_PPO-logging]: No-Op features: avg_mlu={avg_mlu}, total_flows={total_flows}")
            # Insert No-Op at index 0
            # features.append([1 - avg_mlu, avg_flows/max_flows])  # No-Op has average MLU and total flow count
            # node_ids.append('no_op')
            # valid_switches.append(True)  # No-Op is always valid

            for node in switch_nodes:
                mlu = 0.0
                flow_cnt = 0
                actual_flow_count = 0
                
                # Get flow count
                for flow_node in graph_viz['nodes']:
                    if (flow_node.get('type') == 'flow' and
                        flow_node.get('data', {}).get('location') == node.get('data',{}).get('dpid')):
                        actual_flow_count += 1
                
                if head1_data and 'nodes' in head1_data:
                    for head_node in head1_data['nodes']:
                        if head_node.get('id') == node.get('id'):
                            mlu = head_node['data'].get('mlu', 0.0)
                            flow_cnt = head_node['data'].get('flow_count', 0) 
                            break
                
                flow_cnt = max(flow_cnt, actual_flow_count)
                features.append([mlu, flow_cnt])  
                node_ids.append(node['id'])
                valid_switches.append(flow_cnt > 0)
                max_cnt = max(max_cnt, flow_cnt)

            avg_mlu = np.mean([f[0] for f in features]) if features else 0.0
            avg_flows = np.mean([f[1] for f in features]) if features else 1.0
            #max_flows = max([f[1] for f in features], default=1.0)
            max_count = max(max_cnt, 1)  # Avoid division by zero

            no_op_row = [1 - avg_mlu, avg_flows]             # or any summary you like
            features.insert(0, no_op_row)
            node_ids.insert(0, 'no_op')
            valid_switches.insert(0, True)
            
            features = [[mlu, cnt / max_count] for mlu, cnt in features]  # Normalize flow counts

            # LOG.info(f"[H_G_PPO-logging]: Switch features prepared: {features}")
            # LOG.info(f"[H_G_PPO-logging]: Switch node IDs: {node_ids}")
            # Convert to tensor and encode
            x = torch.FloatTensor(features).to(self.device)
            
            # Create edge index with self-loops for all nodes including No-Op
            num_nodes = len(features)
            edge_list = []
            dist = None
            # Add self-loop for No-Op
            edge_list.append([0, 0])
            
            # Extract existing edges and shift indices by 1 (due to No-Op at index 0)
            for edge in graph_viz['edges']:
                if edge.get('edge_type') == 'switch_link' or edge.get('type') == 'switch_link':
                    src_id = edge['from']
                    dst_id = edge['to']
                    # Find indices in our node list (excluding No-Op)
                    if src_id in node_ids[1:] and dst_id in node_ids[1:]:
                        src_idx = node_ids.index(src_id)
                        dst_idx = node_ids.index(dst_id)
                        edge_list.append([src_idx, dst_idx])
            
            if not edge_list:
                edge_list = [[i, i] for i in range(num_nodes)]  # All self-loops
            
            edge_index = torch.LongTensor(edge_list).t().to(self.device)
            
            with torch.no_grad():
                encoded = self._encode_eval(self.switch_encoder, x, edge_index)
                result['encoded_state'] = encoded
                
                logits = self.switch_actor(encoded).squeeze(-1)
                
                # Apply masking for switches with no flows (but not No-Op)
                flow_tensor = torch.tensor([f[1] for f in features], device=self.device)
                #invalid_mask = torch.zeros(len(features), dtype=torch.bool, device=self.device)
                #invalid_mask[1:] = (flow_tensor[1:] == 0)  # Don't mask No-Op
                invalid_mask = (flow_tensor == 0)  # Mask all switches with no flows
                
                if self.config.get('mask_invalid_actions', True):
                    # Check if all real options are invalid
                    logits = self._apply_mask(logits, invalid_mask)
                    if logits is None:
                        result['validation']['error'] = 'All switches masked'
                        return result
                        
                # Add exploration noise
                if self.exploration_rate > self.config['min_exploration']:
                    logits = logits + torch.randn_like(logits) * self.exploration_rate
                
                # Always create distribution and sample
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                action = dist.sample()
                
                # Store transition
                state, value = self._get_state_value(encoded, action, self.switch_critic)
                #local_act, _ = self._localise_action()
                self._store_transition(
                    'switch',
                    encoded,
                    action,
                    dist.log_prob(action),
                    value
                )
                
                selected_idx = action.item()
                result['valid'] = True
                result['selection'] = node_ids[selected_idx]
                result['is_no_op'] = (selected_idx == 0)
                
                if result['is_no_op']:
                    result['validation']['action_type'] = 'no_op'
                    result['validation']['network_mlu'] = avg_mlu
                    result['validation']['total_flows'] = avg_flows
                    result['validation']['selected_mlu'] = avg_mlu
                    result['validation']['selected_flow_count'] = avg_flows
                else:
                    result['validation']['selected_flow_count'] = features[selected_idx][1]
                    result['validation']['selected_mlu'] = features[selected_idx][0]
                
                result['validation']['total_switches'] = len(switch_nodes)
                result['validation']['valid_switches'] = sum(valid_switches[1:])  # Exclude No-Op
            
            return result
            
        except Exception as e:
            return {
                'valid': False,
                'selection': None,
                'validation': {'error': f'Exception in switch selection: {e}'},
                'encoded_state': None,
                'is_no_op': False
            }


    def _select_port_with_validation(self, switch_id, graph_viz, port_data=None):
        """Select a port with No-Op option and proper masking"""
        try:
            result = {
                'valid': False,
                'selection': None,
                'validation': {},
                'encoded_state': None,
                'is_no_op': False
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
            

            high_utilization_threshold = self.config.get('high_utilization_threshold', 0.8)
            max_util_on_sw = max(n['data'].get('utilization', 0.0) for n in port_nodes)
            no_busy_ports = max_util_on_sw < high_utilization_threshold
            # Prepare features
            features = []
            node_ids = []
            valid_ports = []
            
            # Add No-Op option first
            # No-Op features: average utilization and total flow count
            avg_util = np.mean([n['data'].get('utilization', 0.0) for n in port_nodes])
            max_util = max([n['data'].get('utilization', 0.0) for n in port_nodes], default=1.0)
            total_flows = max(np.sum([n['data'].get('flow_count', 0) for n in port_nodes]), 1)
            max_flows = max([n['data'].get('flow_count', 0) for n in port_nodes], default=1.0)
            avg_flows = total_flows / len(port_nodes) if port_nodes else 1.0

            features.append([1 - max_util, avg_flows/max_flows])  # No-Op has average utilization and total flow count
            node_ids.append('no_op')
            valid_ports.append(True)  # No-Op is always valid
            
            # Add actual ports
            for node in port_nodes:
                util = node['data'].get('utilization', 0.0)
                flow_count = node['data'].get('flow_count', 0)
                features.append([util, min(flow_count / max_flows, 1.0)])  # Normalize flow count
                node_ids.append(node['id'])
                valid_ports.append(flow_count > 0)
            #LOG.info(f"[H_G_PPO-logging]: Port features prepared: {features}")
            #LOG.info(f"[H_G_PPO-logging]: Port node IDs: {node_ids}")
            
            # Convert to tensor and encode
            x = torch.FloatTensor(features).to(self.device)
            
            if port_data is not None:
                edge_index = port_data.edge_index.to(self.device) + 1  # Shift indices for No-Op

                no_op_self = torch.zeros((2, 1), dtype=torch.long, device=self.device)
                edge_index = torch.cat([edge_index, no_op_self], dim=1)
            else:
                num_ports = len(features)
                edge_index = torch.LongTensor([[i, i] for i in range(num_ports)]).t().to(self.device)
            
            with torch.no_grad():
                encoded = self._encode_eval(self.port_encoder, x, edge_index)
                result['encoded_state'] = encoded
                
                logits = self.port_actor(encoded).squeeze(-1)
                logits[0] -= 2.0  # Encourage No-Op by default
                
                # Apply masking (but not to No-Op)
                flow_tensor = torch.tensor([f[1] for f in features], device=self.device)
                invalid_mask = torch.zeros(len(features), dtype=torch.bool, device=self.device)
                invalid_mask[1:] = (flow_tensor[1:] == 0)  # Don't mask No-Op
                
                if self.config.get('mask_invalid_actions', True):
                    logits = self._apply_mask(logits, invalid_mask)
                    if logits is None:  # All ports invalid
                        result['validation']['error'] = 'All ports masked due to no flows'
                        return result

                if self.exploration_rate > self.config['min_exploration']:
                    logits = logits + torch.randn_like(logits) * self.exploration_rate

                probs = F.softmax(logits, dim=0)
                dist = Categorical(probs)
                action = dist.sample()
                
                # Store transition
                state, value = self._get_state_value(encoded, action, self.port_critic)
                #local_act, local_lp = self._localise_action()
                self._store_transition(
                    'port',
                    encoded,
                    action,
                    dist.log_prob(action),
                    value
                )
                
                selected_idx = action.item()
                result['valid'] = True
                result['selection'] = node_ids[selected_idx]
                result['is_no_op'] = (selected_idx == 0)
                
                if result['is_no_op']:
                    result['validation']['action_type'] = 'no_op'
                    result['validation']['avg_utilization'] = avg_util
                    result['validation']['total_flows'] = total_flows

                else:
                    result['validation']['selected_flow_count'] = features[selected_idx][1]
                    result['validation']['utilization'] = features[selected_idx][0]
                
                result['validation']['total_ports'] = len(port_nodes)
                result['validation']['valid_ports'] = sum(valid_ports[1:])  # Exclude No-Op
            
            return result
            
        except Exception as e:
            return {
                'valid': False,
                'selection': None,
                'validation': {'error': f'Exception in port selection: {e}'},
                'encoded_state': None,
                'is_no_op': False
            }


    def _select_switch_with_validation1(self, head1_data, graph_viz, switch_data=None):
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
                actual_flow_count = 0
                '''for flow_node in graph_viz['nodes']:
                    if (flow_node.get('type') == 'flow' and
                        flow_node.get('data', {}).get('location') == node.get('data',{}).get('dpid')):
                        actual_flow_count += 1'''

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
                #LOG.info(f"[H_G_PPO-logging]: Actual flow count for switch {node['id']}: {actual_flow_count}")
                #LOG.info(f"[H_G_PPO-logging]: MLU for switch {node['id']}: {mlu}, Flow count: {flow_cnt}")

                flow_cnt = max(flow_cnt, actual_flow_count)  # Ensure flow count is accurate
                #LOG.info(f"[H_G_PPO-logging]: Final flow count for switch {node['id']}: {flow_cnt}")
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
                #local_act, _ = self._localise_action()
                self._store_transition(
                    'switch',
                    encoded,
                    action,
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
            'value': value.cpu().view(1),  # Ensure value is a 1D tensor
            'reward': torch.tensor(penalty, device=self.device).view(1)  # Store penalty as reward
        })

        
        
        if self.logging:
            print(f"[H_G_PPO-logging]: Stored partial failure at {failed_level} with penalty {penalty}")
    
    
    
    

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
    
    # hierarchical_gnn_ppo.py  ─ inside class HierarchicalGNNPPOAgent
    def update(self, rewards: Dict[str, float], new_state=None, no_op_level=None) -> None:
        """
        Attach the *scalar* reward to the **latest** transition of each level,
        decay ε-greedy exploration, and launch PPO when enough samples exist.
        """
        # ── 1. credit assignment  (replace old block) ─────────────────────────
        for lvl, dq in self.memory.items():
            if not dq:
                continue

            r = float(rewards.get(lvl, 0.0))
            reward_tensor = torch.tensor(r, device=self.device).view(1)

            # fill reward into all new transitions since the last update
            for tr in reversed(dq):
                if 'reward' in tr:              # stop at first already-credited sample
                    break
                tr['reward'] = reward_tensor
            if lvl == no_op_level:
                break  # stop at the first level with No-Op


        # ── 2. ε-decay ──────────────────────────────────────────────────────
        self.exploration_rate = max(
            self.exploration_rate * self.config['exploration_decay'],
            self.config['min_exploration']
        )

        # ── 3. train if ready ───────────────────────────────────────────────
        self._ppo_update()

    def update1(self, rewards: Dict[str, float], new_state):
        """Collect reward, adjust exploration, and trigger PPO."""
        LOG.info("[H_G_PPO-logging]: Updating agent with new rewards and state…")
        # store new state
        self.switch_reward_history.append(rewards.get('switch', 0.0))
        self.port_reward_history.append(rewards.get('port', 0.0))
        self.flow_reward_history.append(rewards.get('flow', 0.0))
        self.new_port_reward_history.append(rewards.get('new_port', 0.0))
        #r_sw = r_po = r_fl = r_np = rewards  # default reward for all levels
        r_sw = rewards.get('switch', 0.0)
        r_po = rewards.get('port', 0.0)
        r_fl = rewards.get('flow', 0.0)
        r_np = rewards.get('new_port', 0.0)
        LOG.info(f"[H_G_PPO-logging]: Received rewards: {rewards}")
        LOG.info(f"[H_G_PPO-logging]: Switch reward: {r_sw}, Port reward: {r_po}, Flow reward: {r_fl}, New Port reward: {r_np}")
        # store rewards in memory
        # attach reward to LAST transition of each level
        for lvl, dq in self.memory.items():
            if dq and 'reward' not in dq[-1]:
                r = rewards.get(lvl, 0.0)  # default reward for this level
                #dq[-1]['reward'] = torch.tensor(float(r), device=self.device).view(1)  # store reward
                #if lvl == 'switch': r
                #elif lvl == 'port': r
                #elif lvl == 'flow': r
                #elif lvl == 'new_port': r
                reward_tensor = torch.tensor(float(r), device=self.device).view(1)


                for transition in dq:
                    if 'reward' not in transition:
                        transition['reward'] = reward_tensor

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

    # ------------------------------------------------------------------
    #  Per-level PPO update
    # ------------------------------------------------------------------
    # ▸ NEW VERSION – batched, per-level PPO
    # inside HierarchicalGNNPPOAgent
    def _ppo_update(self) -> None:
        clip   = self.config['clip_epsilon']
        v_coef = self.config['value_loss_coef']

        for lvl in ('switch', 'port', 'flow', 'new_port'):

            buf = [tr for tr in self.memory[lvl] if 'reward' in tr]
            if len(buf) < self.config['min_mem'][lvl]:
                continue

            actor, critic, optim = self.ACTOR[lvl], self.CRITIC[lvl], self.opt[lvl]

            entropies = []

            optim.zero_grad()
            total_loss = torch.tensor(0.0, device=self.device)

            for tr in buf:                                      # ← one by one
                
                enc   = tr['state'].to(self.device)             # (N,H) variable N
                act   = tr['action'].to(self.device).long()     # scalar
                log_b = tr['log_old'].to(self.device)           # (1,)
                ret   = tr['reward'].to(self.device)            # (1,)

                logits = actor(enc).squeeze(-1)                 # (N,)
                dist   = torch.distributions.Categorical(logits=logits)
                log_pi = dist.log_prob(act)                     # (1,)
                entropy= dist.entropy().mean()
                entropies.append(entropy.item())                # collect entropy

                val    = critic(enc)[act].view(1)               # (1,)
                adv    = (ret - val.detach())

                ratio  = torch.exp(log_pi - log_b)              # (1,)
                surr1  = ratio * adv
                surr2  = torch.clamp(ratio, 1-clip, 1+clip) * adv
                actor_L  = -torch.min(surr1, surr2) - self.config['entropy_coef'][lvl]*entropy
                critic_L = 0.5 * (ret - val).pow(2).mean()  # value loss
                loss = actor_L + v_coef * critic_L
                # accumulate loss
                total_loss += loss.squeeze()  # (1,) → scalar
                

            batch_loss = total_loss / len(buf)  # average loss over batch
            batch_loss.backward()  # backpropagate
            torch.nn.utils.clip_grad_norm_(actor.parameters(), self.config['max_grad_norm'])
            torch.nn.utils.clip_grad_norm_(critic.parameters(), self.config['max_grad_norm'])
            optim.step()
            # ---------------- Metrics ----------------------------------------
            mean_entropy = float(np.mean(entropies)) if entropies else 0.0
            self.tb_writer.add_scalar(f'{lvl}/entropy', mean_entropy,
                                       self.training_step)
            self.tb_writer.add_scalar(f'{lvl}/batch_loss',  batch_loss.item(),
                                      self.training_step)
            self.tb_writer.add_scalar(f'{lvl}/actor_loss',  actor_L.item(),
                                      self.training_step)
            self.tb_writer.add_scalar(f'{lvl}/critic_loss', critic_L.item(),
                                      self.training_step)

            # keep CSV/avg bookkeeping
            if not hasattr(self, '_batch_metrics'):
                self._batch_metrics = {l: {'actor': [], 'critic': [], 'entropy': []}
                            for l in ('switch','port','flow','new_port')}

            self._batch_metrics[lvl]['actor'].append(actor_L.item())
            self._batch_metrics[lvl]['critic'].append(critic_L.item())
            self._batch_metrics[lvl]['entropy'].append(mean_entropy)   # NEW

            
            LOG.info(f"[{lvl}] PPO update done – {len(buf)} samples flushed.")
            buf.clear()
            self.memory[lvl] = [tr for tr in self.memory[lvl] if 'reward' not in tr]
            # >>> advance global counter *once per level update*
            self.training_step += 1
            # ------------------------------------------------------------
            # write aggregated metrics once per PPO call
            if hasattr(self, '_batch_metrics') and any(m['actor'] for m in self._batch_metrics.values()):
                actor_losses  = {l: self._batch_metrics[l]['actor']  for l in self._batch_metrics}
                critic_losses = {l: self._batch_metrics[l]['critic'] for l in self._batch_metrics}
                entropy_bonuses = {l: self._batch_metrics[l]['entropy'] for l in self._batch_metrics}
                self._store_training_metrics(actor_losses, critic_losses, entropy_bonuses)
                self._batch_metrics = {l: {'actor': [], 'critic': [], 'entropy': []} for l in self._batch_metrics}



    def _ppo_update111(self) -> None:
        """
        Update each hierarchy level with proper batch processing
        """
        for lvl in ('switch', 'port', 'flow', 'new_port'):
            buf = self.memory[lvl]
            if len(buf) < self.min_mem:
                continue
                
            # Convert buffer to lists for batch processing
            all_transitions = list(buf)
            n_samples = len(all_transitions)
            
            # Process in batches
            for epoch in range(self.config['n_epochs']):
                # Shuffle for each epoch
                indices = torch.randperm(n_samples)
                
                for start_idx in range(0, n_samples, self.config['batch_size']):
                    # Get batch indices
                    batch_indices = indices[start_idx:start_idx + self.config['batch_size']]
                    batch = [all_transitions[i] for i in batch_indices]
                    
                    # Prepare batch tensors
                    states = torch.cat([t['state'] for t in batch]).to(self.device)
                    actions = torch.cat([t['action'] for t in batch]).to(self.device)
                    log_old = torch.cat([t['log_old'] for t in batch]).to(self.device)
                    rewards = torch.cat([t['reward'] for t in batch]).to(self.device)
                    
                    # Forward pass with current parameters
                    logits = self.ACTOR[lvl](states).squeeze(-1)
                    values = self.CRITIC[lvl](states).squeeze(-1)
                    
                    # Calculate losses (rest of the PPO logic remains the same)
                    dist = torch.distributions.Categorical(logits=logits)
                    log_new = dist.log_prob(actions)
                    entropy = dist.entropy().mean()
                    
                    # Advantages
                    adv = rewards - values.detach()
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    
                    # PPO objectives
                    ratio = torch.exp(log_new - log_old)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - self.config['clip_epsilon'], 
                                    1 + self.config['clip_epsilon']) * adv
                    actor_loss = -torch.min(surr1, surr2).mean() - \
                            self.config['entropy_coef'][lvl] * entropy
                    
                    # Value loss with clipping
                    v_clip = values.detach() + torch.clamp(
                        values - values.detach(),
                        -self.config['clip_epsilon'],
                        self.config['clip_epsilon'])
                    critic_loss = 0.5 * torch.max((rewards - values).pow(2),
                                                (rewards - v_clip).pow(2)).mean()
                    
                    loss = actor_loss + self.config['value_loss_coef'] * critic_loss
                    
                    # Optimize
                    self.opt[lvl].zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.opt[lvl].param_groups[0]['params'],
                        self.config['max_grad_norm'])
                    self.opt[lvl].step()
                    
                    # Store metrics for this batch
                    if not hasattr(self, 'batch_metrics'):
                        self.batch_metrics = {l: {'actor': [], 'critic': [], 'entropy': []} 
                                            for l in ['switch', 'port', 'flow', 'new_port']}
                    self.batch_metrics[lvl]['actor'].append(actor_loss.item())
                    self.batch_metrics[lvl]['critic'].append(critic_loss.item())
                    self.batch_metrics[lvl]['entropy'].append(entropy.item())
            
            # Clear buffer after all epochs
            buf.clear()
            
        # Store training metrics after updating all levels
        if hasattr(self, 'batch_metrics'):
            # Ensure all levels are present in the metrics dictionaries
            all_levels = ['switch', 'port', 'flow', 'new_port']
            
            actor_losses = {}
            critic_losses = {}
            entropy_bonuses = {}
            
            for lvl in all_levels:
                if lvl in self.batch_metrics:
                    actor_losses[lvl] = self.batch_metrics[lvl]['actor']
                    critic_losses[lvl] = self.batch_metrics[lvl]['critic']
                    entropy_bonuses[lvl] = self.batch_metrics[lvl]['entropy']
                else:
                    # Provide empty lists for levels that weren't updated
                    actor_losses[lvl] = []
                    critic_losses[lvl] = []
                    entropy_bonuses[lvl] = []
            
            self._store_training_metrics(actor_losses, critic_losses, entropy_bonuses)
            self.batch_metrics = {}  # Clear for next update
    def _ppo_update1(self) -> None:
        """
        Each hierarchy level keeps its own buffer and optimiser.
        We loop over levels; if that buffer has ≥ self.min_mem samples, we:
        • rebuild a fresh forward pass (logits, values) so autograd
            connects to current network parameters;
        • compute PPO‐clip loss (+ optional entropy, value-clip);
        • optimise *only* that level's parameters.
        """
        for lvl in ('switch', 'port', 'flow', 'new_port'):
            buf = self.memory[lvl]
            if len(buf) < self.min_mem:                 # gate
                continue

            # ── 1. gather batch (all tensors are 1-D rows) ────────────────
            states  = torch.cat([t['state']   for t in buf]).to(self.device)  # (B,H)
            actions = torch.cat([t['action']  for t in buf]).to(self.device)  # (B,) all 0
            log_old = torch.cat([t['log_old'] for t in buf]).to(self.device)  # (B,)
            rets    = torch.cat([t['reward']  for t in buf]).to(self.device)  # (B,)
            #  values from buffer are not needed any more

            # ── 2. forward through *current* actor & critic (with grad) ──
            logits = self.ACTOR[lvl](states).squeeze(-1)      # (B,)
            values = self.CRITIC[lvl](states).squeeze(-1)     # (B,)

            dist   = torch.distributions.Categorical(logits=logits)
            log_new = dist.log_prob(actions)                  # (B,)
            entropy = dist.entropy().mean()

            # ── 3. advantages  (simple TD(0); replace by GAE if you like) ─
            adv = rets - values.detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # ── 4. PPO objectives ─────────────────────────────────────────
            ratio = torch.exp(log_new - log_old)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio,
                                1 - self.config['clip_epsilon'],
                                1 + self.config['clip_epsilon']) * adv
            actor_loss = -torch.min(surr1, surr2).mean() \
                        - self.config['entropy_coef'][lvl] * entropy

            # value-clip
            v_clip = values.detach() + torch.clamp(
                        values - values.detach(),
                        -self.config['clip_epsilon'],
                        self.config['clip_epsilon'])
            critic_loss = 0.5 * torch.max((rets - values).pow(2),
                                        (rets - v_clip).pow(2)).mean()

            loss = actor_loss + self.config['value_loss_coef'] * critic_loss

            # ── 5. optimise only this level's parameters ─────────────────
            self.opt[lvl].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.opt[lvl].param_groups[0]['params'],
                self.config['max_grad_norm'])
            self.opt[lvl].step()

            # ── 6. bookkeeping ───────────────────────────────────────────
            self.tb_writer.add_scalar(f'{lvl}/actor_loss',  actor_loss.item(), self.step)
            self.tb_writer.add_scalar(f'{lvl}/critic_loss', critic_loss.item(), self.step)
            buf.clear()
            self.step += 1
            LOG.info(f"[{lvl}] step {self.step:4d}  "
                    f"AL {-actor_loss.item():.3f}  CL {critic_loss.item():.3f}")




    # One optimiser step per level when its buffer is ready.
    # -----------------------------------------------------------
    def _ppo_update1(self) -> None:
        """Update all levels with their own optimisers."""
        LOG.info("[H_G_PPO-logging]: Starting PPO update for all levels…")
        # inside class HierarchicalGNNPPOAgent  – e.g. right after __init__()




        for lvl in ('switch', 'port', 'flow', 'new_port'):
            buf = self.memory[lvl]
            LOG.info(f"[H_G_PPO-logging]: {lvl} buffer size: {len(buf)}")
            if len(buf) < self.min_mem:          # gate
                continue
            LOG.info(f"[H_G_PPO-logging]:Step {self.step}: Updating {lvl} with {len(buf)} transitions…")
            # flat tensors -------------------------------------------------
            states  = torch.cat([t['state']   for t in buf]).to(self.device)
            actions = torch.cat([t['action']  for t in buf]).to(self.device)
            log_old = torch.cat([t['log_old'] for t in buf]).to(self.device)
            rets    = torch.cat([self._as_row(t['reward']) for t in buf]).to(self.device)
            vals    = torch.cat([t['value']   for t in buf]).to(self.device)

            # advantage (GAE-λ optional) -----------------------------------
            adv = (rets - vals)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # single-choice categorical (log_new == 0) ---------------------
            ratio = torch.exp(-log_old)              # log_new = 0
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio,
                                1-self.config['clip_epsilon'],
                                1+self.config['clip_epsilon']) * adv
            actor_loss  = -torch.min(surr1, surr2).mean()

            # value clip ---------------------------------------------------
            v_clipped   = vals + torch.clamp(vals - vals.detach(),
                                             -self.config['clip_epsilon'],
                                             self.config['clip_epsilon'])
            v_loss = 0.5 * torch.max((rets - vals).pow(2),
                                      (rets - v_clipped).pow(2)).mean()

            # optimise -----------------------------------------------------
            self.opt[lvl].zero_grad()
            (actor_loss + self.config['value_loss_coef'] * v_loss).backward()
            nn.utils.clip_grad_norm_(self.opt[lvl].param_groups[0]['params'],
                                     self.config['max_grad_norm'])
            self.opt[lvl].step()

            # housekeeping -------------------------------------------------
            self.tb_writer.add_scalar(f'{lvl}/actor_loss',  actor_loss.item(),
                                      self.step)
            self.tb_writer.add_scalar(f'{lvl}/critic_loss', v_loss.item(),
                                      self.step)
            LOG.info(f"[H_G_PPO-logging]: {lvl} update: actor_loss={actor_loss.item():.4f}, critic_loss={v_loss.item():.4f}")
            buf.clear()
            self.step += 1

    # ------------------------------------------------------------------
    # replace the current placeholder implementation
    # ------------------------------------------------------------------
    def _calculate_gae(self,
                    rewards:  torch.Tensor,   # shape [T]
                    values:   torch.Tensor,   # shape [T]
                    last_val: float = 0.0,    # V(sT) for bootstrap
                    dones:    torch.Tensor = None
                    ) -> torch.Tensor:
        """
        Vectorised GAE(λ) for one trajectory.

        Parameters
        ----------
        rewards  : Tensor [T]      immediate rewards r_t
        values   : Tensor [T]      value estimate V(s_t)
        last_val : float           value for the state *after* the last step
        dones    : Tensor [T]      1 if s_{t+1} is terminal else 0

        Returns
        -------
        advantages : Tensor [T]
        """
        if dones is None:
            # continuing task – never terminal inside the slice
            dones = torch.zeros_like(rewards)

        dones = dones.float()  # ensure dones are float for calculations
        T = rewards.size(0)
        adv = torch.zeros(T, device=rewards.device)
        gae = 0.0
        for t in reversed(range(T)):
            mask      = 1.0 - dones[t]
            delta     = rewards[t] + self.config['gamma'] * last_val * mask - values[t]
            gae       = delta + self.config['gamma'] * self.config['gae_lambda'] * mask * gae
            adv[t]    = gae
            last_val  = values[t]        # V_{t} becomes V_{t+1} in next iteration
        return adv


    
    
    def _calculate_gae1(self, transitions, values):
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
            'optimizer':{
                'switch': self.opt['switch'].state_dict(),
                'port': self.opt['port'].state_dict(),
                'flow': self.opt['flow'].state_dict(),
                'new_port': self.opt['new_port'].state_dict()
            },
            'exploration_rate': self.exploration_rate,
            'config': self.config
        }
        torch.save(checkpoint, path)
        LOG.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model checkpoint"""
        if not os.path.exists(path):
            LOG.error(f"[HGP][Load model] ERROR: Checkpoint file {path} does not exist.")
            return False
        
        try:
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
            if 'optimizer' in checkpoint:
                for lvl in ('switch', 'port', 'flow', 'new_port'):
                    if lvl in checkpoint['optimizer']:
                        self.opt[lvl].load_state_dict(checkpoint['optimizer'][lvl])
            
            
            self.exploration_rate = checkpoint.get('exploration_rate', 0.1)
            
            

            LOG.info(f"Model loaded from {path}")
            
            
            return True
        except Exception as e:
            LOG.error(f"[HGP][Load model] ERROR: Failed to load model from {path}. Exception: {e}")
            return False


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
