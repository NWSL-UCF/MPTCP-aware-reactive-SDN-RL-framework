# agent2_interface.py
"""
Agent2 Interface - Ryu Application for Hierarchical GNN-PPO Agent
Main entry point that interfaces with the SDN controller
"""

import sys
import os
import time
import logging
import statistics
from collections import deque
from ryu.base import app_manager
from ryu.base.app_manager import lookup_service_brick
from ryu.lib import hub
import setting


# Add RL2 to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from RL2.agents.hierarchical_gnn_ppo import HierarchicalGNNPPOAgent
from RL2.config import get_config
from RL2.utils import reward_functions

LOG = logging.getLogger('ryu.app.agent2_interface')


class Agent2Interface(app_manager.RyuApp):
    """
    Ryu application that interfaces Agent2 with the SDN controller
    Manages periodic updates and communication with other modules
    """
    
    def __init__(self, *args, **kwargs):
        super(Agent2Interface, self).__init__(*args, **kwargs)
        self.name = 'Agent2Interface'
        self.enable_logging = setting.AGENT2_LOGGING
        
        # Check if Agent2 is enabled
        self.enabled = setting.USE_AGENT2
        if self.enable_logging:
            LOG.info(f"Agent2 Interface enabled: {self.enabled}")
        if not self.enabled:
            LOG.info("[agent2-interface-logging]: Agent2 is disabled in settings")
            return
        
        # Load configuration
        self.config = get_config()
        self.update_interval = self.config['update_interval']  # 2 minutes
        
        # Initialize agent
        self.agent = HierarchicalGNNPPOAgent(self.config)
        LOG.info(f"Agent2 device: {self.agent.device}")
        
        # Load pre-trained model if specified
        '''if setting.AGENT2_LOAD_MODEL:
            model_path = "models/agent2_model.pt"
            if os.path.exists(model_path):
                try:
                    self.agent.load_model(model_path)
                    LOG.info(f"Loaded Agent2 model from {model_path}")
                except Exception as e:
                    LOG.warning(f"Could not load model: {e}")
            else:
                LOG.info("[agent2-interface-logging]: No pre-trained Agent2 model found")'''
        if setting.AGENT2_LOAD_MODEL:
            self._load_pretrained_model()
        
        # References to other modules
        self.topology_manager = None
        self.multi_head_manager = None
        self.forwarding = None
        self.network_monitor = None
        self.flow_rate_monitor = None
        
        # State tracking
        self.previous_mlu = float('inf')
        self.action_history = deque(maxlen=100)
        self.no_op_history = deque(maxlen=50)
        self.action_time_history = deque(maxlen=10)
        self.mlu_history = deque(maxlen=50)
        self.m_history = deque(maxlen=50)
        self.m_reward_history = deque(maxlen=50)
        self.last_decision_time = 0
        self.m_reward_history.append(0.0)  # Initialize with zero reward
        
        # File naming
        self.file_timestamp = time.strftime("%m-%d_%H-%M")
        self.model_file_name = f"models/agent2/agent2_model_{self.file_timestamp}.pt"
        self.metrics_calculation_filename = f"RL2/data/Metrics/metrics_{self.file_timestamp}.json"

        # Start background tasks
        self.reference_thread = hub.spawn(self._setup_references)
        self.update_thread = hub.spawn(self._periodic_update)
        self._periodic_metrics_calculation = hub.spawn(self._periodic_metrics_calculation)
        if setting.AGENT2_SAVE_MODEL:
            self.save_thread = hub.spawn(self._periodic_save)
        
        LOG.info("[agent2-interface-logging]: Agent2 Interface initialized successfully")
    
    def _load_pretrained_model(self):
        """Load pre-trained model with multiple fallback options"""
        try:
            # Check if specific model name is provided in settings
            model_name = getattr(setting, 'AGENT2_MODEL_NAME', None)
            
            if model_name:
                # Load specific model by name
                model_path = f"models/{model_name}"
                if not model_path.endswith('.pt'):
                    model_path += '.pt'
                    
                if os.path.exists(model_path):
                    LOG.info(f"Loading specified model: {model_path}")
                    if self.agent.load_model(model_path):
                        LOG.info(f"Successfully loaded Agent2 model from {model_path}")
                        return
                    else:
                        LOG.warning(f"Failed to load specified model: {model_path}")
            
            # Fallback to default model paths
            default_paths = [
                "models/agent2_model.pt",
                "models/agent2_model_final.pt",
                f"models/agent2_model_{self.file_timestamp}.pt"
            ]
            
            for path in default_paths:
                if os.path.exists(path):
                    LOG.info(f"Attempting to load model from: {path}")
                    if self.agent.load_model(path):
                        LOG.info(f"Successfully loaded Agent2 model from {path}")
                        return
                    else:
                        LOG.warning(f"Failed to load model from: {path}")
            
            LOG.info("[agent2-interface-logging]: No pre-trained Agent2 model found")
            
        except Exception as e:
            LOG.error(f"Error in model loading: {e}")
    def _get_graph_viz_data(self):
        """Get graph visualization data from topology manager"""
        try:
            if not self.topology_manager:
                LOG.warning("[AGENT2_interface] Warning: TopologyManager not available for graph visualization")
                if self.topology_manager and hasattr(self.topology_manager, 'multi_head_manager'):
                    if self.enable_logging:
                        LOG.info("[agent2-interface-logging]: Using MultiHeadManager for graph visualization")
                    self.multi_head_manager = self.topology_manager.multi_head_manager
                    if self.enable_logging:
                        LOG.info("[agent2-interface-logging]: MultiHeadManager reference obtained from TopologyManager")
                else:
                    LOG.error("[agent2-interface-logging] ERROR: TopologyManager not available for graph visualization")
                    # Return empty graph data if topology manager is not available
                    return {
                        'nodes': [],
                        'edges': [],
                        'error': 'TopologyManager not available'
                    }
            
            # Get topology information using the correct method
            #topology_details = self.topology_manager.get_topology_details()
            # Get multi-head graph data and convert to node/edge format
            multi_head_data = self.multi_head_manager.get_multi_head_data()
            if not multi_head_data or 'error' in multi_head_data:
                LOG.warning("[agent2-interface-logging] Warning: No valid multi-head data returned")
                return {'error': 'No data returned from MultiHeadManager'}

            # FIXED: Don't mix with topology visualization data
            # The multi-head data already contains all necessary nodes and edges
            #return multi_head_data
            graph_viz = self.multi_head_manager.get_graph_viz_data()
            return graph_viz

        except AttributeError as e:
            LOG.error(f"[agent2-interface-logging] Error: {e}")
            return {'error': f'TopologyManager method not available: {e}'}
        except Exception as e:
            LOG.error(f"[agent2-interface-logging] Error: getting graph visualization data: {e}")
            return {
                'error': f'Graph viz error: {e}'
            }

    def _get_basic_topology(self):
        """Get basic topology information as fallback"""
        try:
            nodes = []
            edges = []
            
            # Try to get switches from topology manager directly
            if self.topology_manager and hasattr(self.topology_manager, 'switches'):
                for dpid in self.topology_manager.switches.keys():
                    nodes.append({
                        'id': str(dpid),
                        'type': 'switch',
                        'dpid': dpid,
                        'label': f'S{dpid}',
                        'ports': []
                    })
            
            # Try to get from network monitor if topology manager fails
            elif self.network_monitor and hasattr(self.network_monitor, 'switches'):
                try:
                    switches = self.network_monitor.switches
                    for dpid in switches:
                        nodes.append({
                            'id': str(dpid),
                            'type': 'switch',
                            'dpid': dpid,
                            'label': f'S{dpid}',
                            'ports': []
                        })
                except Exception as e:
                    LOG.debug(f"Error getting switches from network monitor: {e}")
            
            # If we have some nodes, return basic topology
            if nodes:
                return {
                    'nodes': nodes,
                    'edges': edges,
                    'timestamp': time.time(),
                    'source': 'basic_fallback'
                }
            
            return None
            
        except Exception as e:
            LOG.debug(f"Error creating basic topology: {e}")
            return None

    def _setup_references(self):
        """Get references to other modules with retry logic"""
        hub.sleep(5)  # Wait for other modules to initialize
        
        max_attempts = 10
        attempt = 0
        
        while attempt < max_attempts:
            try:
                # Try to get topology manager reference
                if not self.topology_manager:
                    try:
                        # Try different possible names for topology manager
                        for app_name in ['TopologyManagerV2', 'topology_manager', 'TopologyManager']:
                            app = app_manager.lookup_service_brick(app_name)
                            if app:
                                self.topology_manager = app
                                LOG.info(f"Connected to {app_name}")
                                break
                    except Exception as e:
                        LOG.debug(f"Could not connect to topology manager: {e}")
                
                # Try to get multi-head manager reference
                if not self.multi_head_manager:
                    try:
                        # Try different approaches to get multi-head manager
                        if self.topology_manager and hasattr(self.topology_manager, 'multi_head_manager'):
                            self.multi_head_manager = self.topology_manager.multi_head_manager
                            LOG.info("[agent2-interface-logging]: Connected to MultiHeadManager via TopologyManager")
                        else:
                            # Try direct lookup
                            for app_name in ['MultiHeadGraphManager', 'multi_head_manager', 'MultiHeadManager']:
                                app = app_manager.lookup_service_brick(app_name)
                                if app:
                                    self.multi_head_manager = app
                                    LOG.info(f"Connected to {app_name}")
                                    break
                    except Exception as e:
                        LOG.debug(f"Could not connect to multi-head manager: {e}")
                
                # Try to get forwarding reference
                if not self.forwarding:
                    try:
                        for app_name in ['Forwarding', 'forwarding']:
                            app = app_manager.lookup_service_brick(app_name)
                            if app:
                                self.forwarding = app
                                LOG.info(f"Connected to {app_name}")
                                break
                    except Exception as e:
                        LOG.debug(f"Could not connect to forwarding: {e}")
                
                # Try to get network monitor reference
                if not self.network_monitor:
                    try:
                        for app_name in ['NetworkMonitor', 'network_monitor']:
                            app = app_manager.lookup_service_brick(app_name)
                            if app:
                                self.network_monitor = app
                                LOG.info(f"Connected to {app_name}")
                                break
                    except Exception as e:
                        LOG.debug(f"Could not connect to network monitor: {e}")
                if not self.flow_rate_monitor:
                    try:
                        for app_name in ['FlowRateMonitor', 'flow_rate_monitor', 'FlowRateModule']:
                            app = app_manager.lookup_service_brick(app_name)
                            if app:
                                self.flow_rate_monitor = app
                                LOG.info(f"Connected to {app_name}")
                                break
                    except Exception as e:
                        LOG.debug(f"Could not connect to flow rate monitor: {e}")
                # Check if all critical references are obtained
                if all([self.topology_manager, self.network_monitor]):
                    LOG.info("[agent2-interface-logging]: Successfully connected to all critical modules")
                    return
                
                # Log which modules are still missing
                missing = []
                if not self.topology_manager:
                    missing.append("TopologyManager")
                if not self.multi_head_manager:
                    missing.append("MultiHeadManager")
                if not self.forwarding:
                    missing.append("Forwarding")
                if not self.network_monitor:
                    missing.append("NetworkMonitor")
                
                LOG.debug(f"Still missing modules: {missing}")
                
            except Exception as e:
                LOG.error(f"Error getting module references: {e}")
            
            attempt += 1
            hub.sleep(3)
        
        LOG.warning("Could not obtain all module references after maximum attempts")
        
        # Log final status
        status = {
            'topology_manager': bool(self.topology_manager),
            'multi_head_manager': bool(self.multi_head_manager),
            'forwarding': bool(self.forwarding),
            'network_monitor': bool(self.network_monitor)
        }
        LOG.info(f"Final module reference status: {status}")
    
    def _periodic_update(self):
        """Main decision-making loop"""
        if self.enable_logging:
            LOG.info("[agent2-interface-logging]: Starting periodic update loop")
        hub.sleep(30)  # Initial delay
        if self.enable_logging:
            LOG.info("[agent2-interface-logging]: Periodic update loop started after initial delay")

        while self.enabled:
            try:
                # Check if enough time has passed
                current_time = time.time()
                if current_time - self.last_decision_time < self.update_interval:
                    LOG.debug("[Agent2_interface] Waiting for next update interval...")
                    hub.sleep(10)
                    continue
                
                # Check if we have necessary references
                if not all([self.multi_head_manager, self.network_monitor]):
                    LOG.debug("[Agent2_interface] Waiting for module references...")
                    if self.enable_logging:
                        LOG.warning("[Agent2_interface] WARNING: MultiHeadManager or NetworkMonitor not available, waiting...")
                    hub.sleep(10)
                    continue
                
                # Get multi-head graph data with validation
                graph_data = self._get_graph_data()
                if not self._validate_graph_data(graph_data):
                    LOG.warning("[Agent2_interface] WARNING: Invalid or incomplete graph data received")
                    hub.sleep(10)
                    continue
                
                # Get current network state with validation
                current_state = self._get_network_state()
                if not self._validate_network_state(current_state):
                    LOG.warning("[Agent2_interface] WARNING: Invalid network state received")
                    hub.sleep(10)
                    continue
                
                current_mlu = current_state['mlu']
                old_link_utilizations = self.network_monitor.get_all_links_utilization()
                Old_m = self._calculate_metric_m(old_link_utilizations)
                
                if self.enable_logging:
                    LOG.info(f"[Agent2_interface] Current MLU: {current_mlu}, Previous MLU: {self.previous_mlu}")
                if not self._is_active_flows():
                    LOG.info("[Agent2_interface] No active MPTCP flows detected, skipping action selection")
                    hub.sleep(10)
                    continue
                # Make hierarchical decision with timeout
                LOG.info("[Agent2_interface] Selecting action with timeout protection")
                #LOG.info(f'[Agent2_interface] Graph data: {graph_data}')
                start_time = time.time()
                action = self._select_action_with_timeout(graph_data, current_state, timeout=60)
                end_time = time.time()
                LOG.info(f"[Agent2_interface] Action selection took {end_time - start_time:.2f} seconds")
                

                if action.get('no_action_needed', False):
                    LOG.warning("[Agent2_interface] No action needed at this time")
                    self.last_decision_time = current_time
                    hub.sleep(10)
                    continue # skip to next iteration
                if self.enable_logging:
                    LOG.info(f"[Agent2_interface] Selected action: {action}")
                if self._validate_action(action): 
                    self.previous_mlu = current_mlu # Update previous MLU before action
                    if self.enable_logging:
                        LOG.info(f"[Agent2_interface] Agent2 Decision: Switch={action['switch']}, "
                                 f"Flow={action['flow']}, NewPort={action['new_port']}")
                    
                    # Execute action with retry logic
                    success = self._execute_action_with_retry(action, max_retries=3)
                    end_time = time.time()
                    LOG.info(f"[Agent2_interface] Action execution took {end_time - start_time:.2f} seconds")
                    if self.enable_logging:
                        LOG.info(f"[Agent2_interface] Action execution success: {success}")

                    if success['status']:
                        # Wait for network to stabilize with timeout
                        stabilization_success = self._wait_for_stabilization(timeout=60) # should be 90

                        if stabilization_success:
                            # Get new state
                            new_state = self._get_network_state() # return new_state, max_mlu, switch_mlus, timestamp
                            LOG.info(f"[Agent2_interface] New network state after action: {new_state}")
                            if self._validate_network_state(new_state):
                                new_mlu = new_state['mlu']
    
                                new_link_utilizations = self.network_monitor.get_all_links_utilization()
                                
                                new_m = self._calculate_metric_m(new_link_utilizations)
                                if action.get('is_no_op', False):
                                    no_op_level = action.get('no_op_level', None)

                                    
                                    
                                    validation = action.get('validation', {})
                                    # Handle No-Op rewards based on level
                                    if no_op_level == 'switch':
                                        r_sw = reward_functions.switch_reward(self.previous_mlu, new_mlu, Old_m, new_m)
                                        r_port = r_flow = r_np = 0.0  # No-Op on switch means no port/flow/new_port action
                                    elif no_op_level == 'port':
                                        #r_sw   = reward_functions.switch_reward(self.previous_mlu, new_mlu)
                                        r_sw = reward_functions.switch_reward(self.previous_mlu, new_mlu, Old_m, new_m)
                                        r_port = reward_functions.port_reward(validation.get('port', {}).get('utilization', 0.0), self.previous_mlu)
                                        r_flow = r_np = 0.0  # No-Op on port means no flow/new_port action
                                    elif no_op_level == 'flow':
                                        #r_sw   = reward_functions.switch_reward(self.previous_mlu, new_mlu)
                                        r_sw = reward_functions.switch_reward(self.previous_mlu, new_mlu, Old_m, new_m)
                                        sel_port_util = validation.get('port', {}).get('utilization', 0.0)
                                        r_port = reward_functions.port_reward(sel_port_util, self.previous_mlu)
                                        r_flow = reward_functions.flow_reward(validation.get('flow', {}).get('selected_flow_rate', 0.0), setting.LINK_MAX_CAPACITY, True)
                                        r_np = 0.0  # No-Op on flow means no port/new_port action
                                    elif no_op_level == 'new_port':
                                        #r_sw   = reward_functions.switch_reward(self.previous_mlu, new_mlu)
                                        r_sw = reward_functions.switch_reward(self.previous_mlu, new_mlu, Old_m, new_m)
                                        sel_port_util = validation.get('port', {}).get('utilization', 0.0)
                                        r_port = reward_functions.port_reward(sel_port_util, self.previous_mlu)
                                        flow_rate = validation.get('flow', {}).get('selected_flow_rate', 0.0)
                                        r_flow = reward_functions.flow_reward(flow_rate, setting.LINK_MAX_CAPACITY, True)
                                        target_util_before = validation.get('new_port', {}).get('selected_utilization', 0.0)
                                        dpid = action.get('switch', None)
                                        dpid = dpid.split('_')[-1]
                                        dpid_port = action.get('port', None)
                                        dpid_port = dpid_port.split('-')[-1]
                                        # from dpid_port remove 'port'
                                        if dpid_port and dpid_port.startswith('port'):
                                            dpid_port = dpid_port.replace('port', '')
                                        target_util_after = self.network_monitor.get_port_utilization(int(dpid), int(dpid_port))
                                        r_np = reward_functions.newport_reward(sel_port_util, target_util_after)

                                    LOG.info(f"[Agent2_interface] No-Op rewards - Level: {no_op_level}, "
                                                f"Switch: {r_sw:.3f}, Port: {r_port:.3f}, "
                                                f"Flow: {r_flow:.3f}, NewPort: {r_np:.3f}")
                                    
                                    
                                    
                                    '''r_sw   = reward_functions.switch_reward(self.previous_mlu, new_mlu)
                                    r_port = reward_functions.port_reward(sel_port_util, self.previous_mlu)
                                    r_flow = reward_functions.flow_reward(flow_rate, setting.LINK_MAX_CAPACITY, dst_reachable)
                                    r_np   = reward_functions.newport_reward(target_util_before,
                                                                            target_util_after)'''


                                else:
                                    validation = action.get('validation', {})
                                    dpid = action.get('switch', None)
                                    dpid = dpid.split('_')[-1]
                                    dpid_port = action.get('port', None)
                                    dpid_port = dpid_port.split('-')[-1]
                                    # from dpid_port remove 'port'
                                    if dpid_port and dpid_port.startswith('port'):
                                        dpid_port = dpid_port.replace('port', '')

                                    LOG.info(f"[Agent2_interface] New MLU after action: {self.previous_mlu} -> {new_mlu}")
                                    #LOG.info(f"[Agent2_interface] action: {action}, ")
                                    #LOG.info(f"[Agent2_interface] success: {success}, ")
                                    LOG.info(f"[Agent2_interface] Validation data: {validation}")
                                    # Calculate reward with error handling
                                    # reward = self._calculate_reward_safe(current_mlu, new_mlu, action)
                                    # port utilization
                                    sel_port_util = validation.get('port', {}).get('utilization', 0.0)
                                    LOG.info(f"[Agent2_interface] Selected port utilization: {sel_port_util}")
                                    # flow rate
                                    flow_rate = validation.get('flow', {}).get('selected_flow_rate', 0.0)
                                    LOG.info(f"[Agent2_interface] Selected flow rate: {flow_rate}")
                                    # destination reachability
                                    dst_reachable = success.get('validate', False)
                                    LOG.info(f"[Agent2_interface] Destination reachable: {dst_reachable}")
                                    target_util_before = validation.get('new_port', {}).get('selected_utilization', 0.0)
                                    LOG.info(f"[Agent2_interface] Target utilization before action: {target_util_before}")
                                    #target_util_after = action.get('target_util_after', 0.0) # need to get it from link_port_utilization
                                    target_util_after = self.network_monitor.get_port_utilization(int(dpid), int(dpid_port))
                                    LOG.info(f"[Agent2_interface] Target utilization after action: {target_util_after}")

                                    #r_sw   = reward_functions.switch_reward(self.previous_mlu, new_mlu)
                                    r_sw = reward_functions.switch_reward(self.previous_mlu, new_mlu, Old_m, new_m)
                                    r_port = reward_functions.port_reward(sel_port_util, self.previous_mlu)
                                    r_flow = reward_functions.flow_reward(flow_rate, setting.LINK_MAX_CAPACITY, dst_reachable)
                                    r_np   = reward_functions.newport_reward(sel_port_util, target_util_before, target_util_after)

                                reward = {
                                    'switch': r_sw,
                                    'port': r_port,
                                    'flow': r_flow,
                                    'new_port': r_np
                                }
                                LOG.info(f"[Agent2_interface] Calculated reward: {reward}")

                                self.m_reward_history.append(r_sw)
                                # Update agent with error handling
                                self._update_agent_safe(reward, new_state)
                                
                                # Log action
                                self._log_action(action, current_mlu, new_mlu, reward)
                                
                                # Update state
                                self.previous_mlu = new_mlu
                                self.last_decision_time = current_time
                            else:
                                LOG.warning("Invalid new state after action execution")
                        else:
                            LOG.warning("Network did not stabilize after action execution")
                    else:
                        LOG.warning("Failed to execute action after retries")
                else:
                    LOG.debug("No valid action selected or action validation failed")
            
            except Exception as e:
                LOG.error(f"Error in periodic update: {e}")
                import traceback
                LOG.error(traceback.format_exc())
                # Add exponential backoff on repeated errors
                self._handle_periodic_update_error()
        
        hub.sleep(30)  # Check every 30 seconds


    def _calculate_metric_m(self, link_utilizations):
        """Calculate MLU from link utilizations"""
        if not link_utilizations:
            return 0.0
        try:
            # Calculate MLU as the average of all link utilizations
            Max_link_utilization = max(link_utilizations)
            avg_utilization = statistics.mean(link_utilizations)
            m = (1- avg_utilization) * Max_link_utilization
            return m
        except statistics.StatisticsError:
            LOG.warning("StatisticsError: No valid link utilizations to calculate MLU")
            return 0.0
        except Exception as e:
            LOG.error(f"Error calculating MLU: {e}")
            return 0.0
        



    def _validate_graph_data(self, graph_data):
        """Validate graph data structure and content"""
        if not graph_data:
            return False
        #if self.enable_logging:
            #LOG.info("[agent2-interface-logging]: Validating graph data structure")
            #LOG.info(f"[agent2-interface-logging]: Graph data keys: {list(graph_data.keys())}")
            #LOG.info(f"[agent2-interface-logging]: Graph data content: {graph_data}")
        if 'error' in graph_data:
            LOG.error(f"[agent2_interface] Error: Graph data contains error: {graph_data['error']}")
            return False
        
        # Check required fields
        required_fields = ['graph_viz']
        for field in required_fields:
            if field not in graph_data:
                LOG.warning(f"[agent2_interface] Warning: Missing required field in graph data: {field}")
                return False
        
        # Validate graph_viz structure
        graph_viz = graph_data['graph_viz']
        if not isinstance(graph_viz, dict):
            LOG.warning("[agent2_interface] Warning: graph_viz is not a dictionary")
            return False
        
        if 'nodes' not in graph_viz or 'edges' not in graph_viz:
            LOG.warning("[agent2_interface] Warning: graph_viz missing nodes or edges")
            return False
        
        if not isinstance(graph_viz['nodes'], list) or not isinstance(graph_viz['edges'], list):
            LOG.warning("[agent2_interface] Warning: graph_viz nodes or edges are not lists")
            return False

        # Check if we have at least some nodes
        if len(graph_viz['nodes']) == 0:
            LOG.warning("[agent2_interface] Warning: No nodes in graph data")
            return False
        
        return True
    
    def _validate_network_state(self, state):
        """Validate network state structure and values"""
        if not isinstance(state, dict):
            return False
        
        required_fields = ['mlu', 'switch_mlus', 'timestamp']
        for field in required_fields:
            if field not in state:
                LOG.warning(f"Missing required field in network state: {field}")
                return False
        
        # Validate MLU value
        mlu = state['mlu']
        if not isinstance(mlu, (int, float)) or mlu < 0:
            LOG.warning(f"Invalid MLU value: {mlu}")
            return False
        
        # Validate timestamp
        timestamp = state['timestamp']
        current_time = time.time()
        if not isinstance(timestamp, (int, float)) or abs(current_time - timestamp) > 300:
            LOG.warning(f"Invalid or stale timestamp: {timestamp}")
            return False
        
        return True

    def _validate_action(self, action):
        """Validate action structure and values"""
        if not action:
            return False
        
        if not isinstance(action, dict):
            LOG.warning("[Agent2 Interface][Validate Action] WARN: Action is not a dictionary")
            return False
        
        # Check if action is marked as valid
        if not action.get('valid', False):
            return False
        
        # Check required fields
        if action.get('is_no_op', False):
            required_fields = ['switch','no_op_level']
            for field in required_fields:
                if field not in action:
                    LOG.warning(f"[Agent2 Interface][Validate Action] WARN: Missing required field in No-Op action: {field}")
                    return False
            return True  # No-Op actions are valid if they have the required fields

        required_fields = ['switch', 'flow', 'new_port']
        for field in required_fields:
            if field not in action:
                LOG.warning(f"[Agent2 Interface][Validate Action] WARN: Missing required field in action: {field}")
                return False
        
        # Validate field types and values
        if not isinstance(action['switch'], (int, str)):
            LOG.warning("[Agent2 Interface][Validate Action] WARN: Invalid switch ID type")
            return False
        
        if not isinstance(action['flow'], (int, str)):
            LOG.warning("[Agent2 Interface][Validate Action] WARN: Invalid flow ID type")
            return False
        
        if not isinstance(action['new_port'], (int, str)):
            LOG.warning("[Agent2 Interface][Validate Action] WARN: Invalid port ID type")
            return False
        
        return True

    def _select_action_with_timeout(self, graph_data, current_state, timeout=30):
        """Select action with timeout protection"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Action selection timed out")
        
        try:
            # Set timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            # Make decision
            action = self.agent.select_action(graph_data, current_state)
            
            # Clear timeout
            signal.alarm(0)
            return action
        
        except TimeoutError:
            LOG.error(f"Action selection timed out after {timeout} seconds")
            return None
        except Exception as e:
            LOG.error(f"Error in action selection: {e}")
            return None
        finally:
            signal.alarm(0)

    def _execute_action_with_retry(self, action, max_retries=3):
        """Execute action with retry logic"""
        for attempt in range(max_retries):
            try:
                success = self._execute_action(action)
                if success['status']:
                    return success
            
                LOG.warning(f"Action execution failed, attempt {attempt + 1}/{max_retries}")
                hub.sleep(2)  # Wait before retry
            
            except Exception as e:
                LOG.error(f"Error executing action (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    hub.sleep(2)
        
        LOG.error(f"Failed to execute action after {max_retries} attempts")
        return {'status': False, 'message': 'Action execution failed after retries'}

    def _wait_for_stabilization(self, timeout=60):
        """Wait for network to stabilize with timeout"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            hub.sleep(1)
            
            # Check if network has stabilized
            # This could involve checking for consistent MLU readings
            try:
                current_state = self._get_network_state()
                if self._validate_network_state(current_state):
                    # Add your stabilization criteria here
                    # For now, just wait the full stabilization time
                    if time.time() - start_time >= 59:  # Minimum stabilization time
                        return True
            except Exception as e:
                LOG.warning(f"Error checking stabilization: {e}")
        
        LOG.warning(f"Network stabilization timed out after {timeout} seconds")
        return False

    def _calculate_reward_safe(self, current_mlu, new_mlu, action):
        """Calculate reward with error handling"""
        try:
            reward = self.agent.calculate_reward(current_mlu, new_mlu, action)
            
            # Validate reward value
            if not isinstance(reward, (int, float)):
                LOG.warning(f"Invalid reward type: {type(reward)}")
                return 0.0
            
            if abs(reward) > 1000:  # Sanity check for extreme values
                LOG.warning(f"Extreme reward value: {reward}")
                return max(-1000, min(1000, reward))  # Clamp
            
            return reward
            
        except Exception as e:
            LOG.error(f"Error calculating reward: {e}")
            # Return simple reward based on MLU improvement
            return float(current_mlu - new_mlu)

    def _update_agent_safe(self, reward, new_state, no_op_level=None):
        """Update agent with error handling"""
        try:
            
            full_rewards = {
                'switch': reward.get('switch', 0.0),
                'port': reward.get('port', 0.0),
                'flow': reward.get('flow', 0.0),
                'new_port': reward.get('new_port', 0.0)
            }

            LOG.info(f"Updating agent with reward: {full_rewards}, new state: {new_state}")
            self.agent.update(full_rewards, new_state, no_op_level=no_op_level)

            total_reward = sum(full_rewards.values())
            if hasattr(self.agent, 'reward_history'):
                self.agent.reward_history.append(total_reward)
                
            self.agent.training_step += 1


        except Exception as e:
            LOG.error(f"Error updating agent: {e}")
            import traceback
            traceback.print_exc()
            # Continue without updating - don't crash the main loop

    def _handle_periodic_update_error(self):
        """Handle repeated errors in periodic update with exponential backoff"""
        if not hasattr(self, '_error_count'):
            self._error_count = 0
            self._last_error_time = 0
        
        current_time = time.time()
        
        # Reset error count if enough time has passed
        if current_time - self._last_error_time > 300:  # 5 minutes
            self._error_count = 0
        
        self._error_count += 1
        self._last_error_time = current_time
        
        # Exponential backoff
        backoff_time = min(300, 5 * (2 ** min(self._error_count - 1, 6)))  # Max 5 minutes
        LOG.warning(f"Error count: {self._error_count}, backing off for {backoff_time} seconds")
        hub.sleep(backoff_time)

    def _get_graph_data(self):
        """Get multi-head graph data with enhanced error handling"""
        try:
            if not self.multi_head_manager:
                return {'error': 'MultiHeadManager not available'}
            

            # Force graph data update before getting data
            self.multi_head_manager.update_graph()
            
            # Get the full multi-head graph data with timeout
            data = self.multi_head_manager.get_multi_head_data()
            pyg = self.multi_head_manager.get_multi_head_pyg_data() 
            if not data:
                return {'error': 'No data returned from MultiHeadManager'}
            
            # Get the graph data in format suitable for visualization
            graph_viz_data = self._get_graph_viz_data()
            
            # Combine both
            data['graph_viz'] = graph_viz_data
            data['pyg_data'] = pyg
            
            return data
            
        except AttributeError as e:
            LOG.error(f"MultiHeadManager method not available: {e}")
            return {'error': f'Method not available: {e}'}
        except Exception as e:
            LOG.error(f"Error getting graph data: {e}")
            return {'error': f'Graph data error: {e}'}

    def _get_network_state(self):
        """Get current network state metrics with enhanced error handling"""
        # Initialize with safe defaults
        max_mlu = 0.0
        switch_mlus = {}
        state = {
            'mlu': max_mlu,
            'switch_mlus': switch_mlus,
            'timestamp': time.time(),
            'mlu_history': list(self.mlu_history)[-10:], # Last 10 MLU values
            'time_since_last_action': (time.time() - self.action_time_history[-1]) if self.action_time_history else float('inf')
        }
        try:
            

            try:
                result = self._calculate_all_switch_mlus()
                if result and isinstance(result, dict):
                    #LOG.info(f"[agent2-interface-logging][GetNetworkState] Calculated all switch MLUs: {result}")
                    max_mlu = result.get('mlu', 0.0)
                    switch_mlus = result.get('switch_mlus', {})
                    state['mlu'] = max_mlu
                    state['switch_mlus'] = switch_mlus
                    # state['timestamp'] = result.get('timestamp', time.time())
            except Exception as e:
                LOG.warning(f"Error calculating all switch MLUs: {e}")
                state['error'] = f'Error calculating all switch MLUs: {e}'
                return state
            '''
            if not self.network_monitor:
                LOG.warning("NetworkMonitor not available")
                state['error'] = 'NetworkMonitor not available'
                return state


            
            # Also check switch-specific MLU from multi-head graph
            if self.multi_head_manager:
                try:
                    if hasattr(self.multi_head_manager, 'switch_mlu_cache'):
                        cache = self.multi_head_manager.switch_mlu_cache
                        if isinstance(cache, dict):
                            for dpid, mlu in cache.items():
                                try:
                                    if isinstance(mlu, (int, float)) and mlu >= 0:
                                        switch_mlus[dpid] = mlu
                                        max_mlu = max(max_mlu, mlu)
                                except (TypeError, ValueError):
                                    continue
                except Exception as e:
                    LOG.debug(f"Error accessing switch MLU cache: {e}")
                state['mlu'] = max_mlu
                self.mlu_history.append(max_mlu)
                state['switch_mlus'] = switch_mlus
                #LOG.info(f"[agent2-interface-logging][GetNetworkState] no_op_history length: {len(self.no_op_history) if self.no_op_history else 0}")
                # Fix the indexing error - use length check and proper indexing
                '''
            if self.no_op_history and len(self.no_op_history) > 0:
                state['recent_no_op_count'] = self.no_op_history[-1].get('recent_no_ops', 0)
            else:
                state['recent_no_op_count'] = 0
            self.mlu_history.append(max_mlu)
            return state

        except Exception as e:
            LOG.error(f"Error getting network state: {e}")
            state['error'] = f'Network state error: {e}'
            # Return state with error
            return state

    def _calculate_all_switch_mlus(self):
        """ Calculate maximum MLU and all MLUs for all switches
        return {
            'mlu': max_mlu,
            'switch_mlus': switch_mlus,
            'timestamp': time.time()
        }
        """
        try:
            if not self.topology_manager:
                LOG.warning("TopologyManager not available for calculating switch MLUs")
                return {
                    'mlu': 0.0,
                    'switch_mlus': {},
                    'timestamp': time.time(),
                    'error': 'TopologyManager not available'
                }
            
            # Get all active switches
            switches = self.topology_manager.switches
            if not switches:
                LOG.warning("No active switches found in topology manager")
                return {
                    'mlu': 0.0,
                    'switch_mlus': {},
                    'timestamp': time.time(),
                    'error': 'No active switches found'
                }
            
            max_mlu = 0.0
            switch_mlus = {}
            
            for switch_dpid in switches:
                mlu = self._calculate_switch_mlu(switch_dpid)
                if mlu >= 0:  # Only consider valid MLUs
                    switch_mlus[switch_dpid] = mlu
                    max_mlu = max(max_mlu, mlu)
            
            return {
                'mlu': max_mlu,
                'switch_mlus': switch_mlus,
                'timestamp': time.time()
            }
        
        except Exception as e:
            LOG.error(f"Error calculating all switch MLUs: {e}")
            return {
                'mlu': 0.0,
                'switch_mlus': {},
                'timestamp': time.time(),
                'error': f'Switch MLU calculation error: {e}'
            }

    def _calculate_switch_mlu(self, switch_dpid):
        """Calculate Maximum Link Utilization for a switch"""
        try:
            
                        
            # Get switch ports
            if not hasattr(self.topology_manager, 'get_all_active_ports_table'):
                return 0.0
            
            #ports = self.topology_manager.switch_all_ports_table.get(switch_dpid, set())
            ports = self.topology_manager.get_all_active_ports_table(switch_dpid)
            if not ports:
                return 0.0
            
            max_utilization = 0.0
            utilization = 0.0
            for port_no in ports:
                if self.network_monitor:
                    # Use network monitor module to get port utilization
                    utilization = self.network_monitor.get_port_utilization(switch_dpid, port_no)

                #utilization = self._get_port_utilization(switch_dpid, port_no)
                if self.enable_logging:
                    #print(f"[agent2 interface]: Port {port_no} utilization: {utilization}")
                    pass
                max_utilization = max(max_utilization, utilization)
            
            # Cache the result
            #self.switch_mlu_cache[switch_dpid] = max_utilization
            if self.enable_logging:
                #print(f"[agent2 interface]: Switch {switch_dpid} MLU: {max_utilization}")
                pass
            return max_utilization
            
        except Exception as e:
            LOG.error(f"Error calculating switch MLU for {switch_dpid}: {e}")
            return 0.0
        

    def get_status(self):
        """Get agent status for API with enhanced error handling"""
        try:
            # Safely check agent device
            device_str = 'not initialized'
            if self.agent:
                try:
                    device_str = str(self.agent.device)
                except Exception:
                    device_str = 'device unavailable'
            
            # Safely check module readiness
            ready = False
            try:
                ready = all([self.multi_head_manager, self.network_monitor])
            except Exception:
                ready = False
            
            # Safely get action count
            actions_taken = 0
            try:
                actions_taken = len(self.action_history)
            except Exception:
                actions_taken = 0
            
            return {
                'enabled': getattr(self, 'enabled', False),
                'ready': ready,
                'actions_taken': actions_taken,
                'current_mlu': getattr(self, 'previous_mlu', float('inf')),
                'last_decision': getattr(self, 'last_decision_time', 0),
                'device': device_str,
                'error_count': getattr(self, '_error_count', 0)
            }
            
        except Exception as e:
            LOG.error(f"Error getting status: {e}")
            return {
                'enabled': False,
                'ready': False,
                'actions_taken': 0,
                'current_mlu': float('inf'),
                'last_decision': 0,
                'device': 'error',
                'error': str(e)
            }

    def _cleanup(self):
        """Cleanup on shutdown with enhanced error handling"""
        LOG.info("[agent2-interface-logging]: Starting Agent2 Interface cleanup")
        
        # Kill threads safely
        threads = ['update_thread', 'save_thread', 'reference_thread']
        if setting.AGENT2_SAVE_MODEL:
            threads.append('save_model_thread')

        for thread_name in threads:
            if hasattr(self, thread_name):
                try:
                    thread = getattr(self, thread_name)
                    if thread:
                        hub.kill(thread)
                        LOG.debug(f"Killed {thread_name}")
                except Exception as e:
                    LOG.warning(f"Error killing {thread_name}: {e}")
        
        # Save final model safely
        if self.agent and setting.AGENT2_SAVE_MODEL:
            try:
                os.makedirs("models/agent2", exist_ok=True)

                self.agent.save_model(f"models/agent2/agent2_model_final_{self.file_timestamp}.pt")
                LOG.info("[agent2-interface-logging]: Saved final Agent2 model")
            except Exception as e:
                LOG.error(f"Error saving final model: {e}")
        
        # Save final action history
        try:
            self._save_action_history()
            LOG.info("[agent2-interface-logging]: Saved final action history")
        except Exception as e:
            LOG.warning(f"Error saving final action history: {e}")
        
        LOG.info("[agent2-interface-logging]: Agent2 Interface cleaned up")

    def _execute_action(self, action):
        """Execute the selected action through forwarding module"""
        current_time = time.time()
        if not self.forwarding:
            LOG.error("Forwarding module not available")
            return {'status': False, 'message': 'Forwarding module not available'}
        
        try:
            # Check if action is No-Op
            if action.get('is_no_op', False):
                recent_no_ops = sum(1 for h in self.no_op_history if current_time - h['timestamp'] < 300)  # 5 minutes
                self.no_op_history.append({
                    'timestamp': current_time,
                    'action': action,
                    'no_op_level': action.get('no_op_level', 'unknown'),
                    'mlu': self.previous_mlu,
                    'recent_no_ops': recent_no_ops
                })
                LOG.info(f"[agent2-interface-logging]: No-Op selected at {action.get('no_op_level')} level")
                LOG.info(f"[agent2-interface-logging]: Recent No-Op count: {recent_no_ops}")
                #LOG.info(f"[agent2-interface-logging]: No-Op history: {self.no_op_history[-10:]}")  # Log last 10 entries

                if recent_no_ops > 10:
                    LOG.warning("[agent2-interface-logging]: WARN: Too many No-Op actions in the last 5 minutes, "
                                "consider reviewing the action selection logic")
            
                
                return {'status': True, 
                        'message': f'No-Op action at {action.get("no_op_level")} level',
                        'validate': True,
                        'is_no_op': True,
                        'action': action,
                        'no_op_level': action.get('no_op_level', 'unknown')
                        }
            # If not a No-Op, proceed with normal action execution
            self.action_time_history.append(current_time)

            # Call forwarding to migrate the flow
            # The action contains: switch, flow, new_port
            success = self.migrate_flow(
                action['switch'],
                action['flow'], 
                action['new_port']
            )
            if success['status']:
                # Update the forwarding module with new flow
                LOG.info(f"[agent2-interface-logging]: Successfully migrated flow {action['flow']} "
                         f"to new port {action['new_port']} on switch {action['switch']}"
                         f" with path {success.get('path', None)}"
                         f" and path_id {success.get('path_id', None)}")
                if not success.get('validate', False):
                    LOG.info(f"[agent2-interface-logging]: Flow migration validated successfully")
                    return success
                if success.get('path_id', None) is None:
                    LOG.warning(f"[agent2-interface-logging]: No path_id returned, "
                                f"using default path_id for flow migration")
                    return {'status': False, 'validate': False, 'message': 'No path_id returned'}
                if success.get('path', None) is None:
                    LOG.warning(f"[agent2-interface-logging]: No path returned, "
                                f"using default path for flow migration")
                    return {'status': False, 'validate': False, 'message': 'No path returned'}
                self.forwarding.agent2_update_flow(
                    action['switch'], 
                    action['flow'], 
                    action['new_port'],
                    success.get('path', None),
                    success.get('path_id', None)
                )
            
            if self.enable_logging:
                LOG.info(f"[agent2-interface-logging]: Flow migration "
                        f"{'successful' if success else 'failed'}")
            
            return success
            
        except Exception as e:
            LOG.error(f"Error executing action: {e}")
            return {'status': False, 'message': str(e)}

    def _log_action(self, action, prev_mlu, new_mlu, reward):
        """Log action details for analysis"""
        action_log = {
            'timestamp': time.time(),
            'action': action,
            'prev_mlu': prev_mlu,
            'new_mlu': new_mlu,
            'reward': reward,
            'mlu_improvement': prev_mlu - new_mlu
        }
        
        self.action_history.append(action_log)
        
        # Save to file periodically
        if len(self.action_history) % 50 == 0:
            self._save_action_history()

    def _save_action_history(self):
        """Save action history to file"""
        try:
            os.makedirs("logs/action_log", exist_ok=True)
            filename = f"logs/action_log/agent2_actions_{self.file_timestamp}.json"
            
            with open(filename, 'w') as f:
                import json
                json.dump(list(self.action_history), f, indent=2)
                
            LOG.debug(f"Saved action history to {filename}")
            
        except Exception as e:
            LOG.warning(f"Could not save action history: {e}")

    def _periodic_save(self):
        """Periodically save model and metrics"""
        hub.sleep(60)  # Initial delay
        
        while self.enabled and setting.AGENT2_SAVE_MODEL:
            try:
                # Save model
                os.makedirs("models/agent2", exist_ok=True)
                self.agent.save_model(self.model_file_name)
                
                # Save training metrics if available
                if hasattr(self.agent, 'training_metrics'):
                    self._save_training_metrics()
                
                LOG.info(f"[agent2-interface-logging]: Saved model and metrics")
                
            except Exception as e:
                LOG.error(f"Error in periodic save: {e}")
            
            hub.sleep(self.config['save_interval'])

    def _save_training_metrics(self):
        """Save training metrics to file"""
        try:
            os.makedirs("logs", exist_ok=True)
            filename = f"logs/agent2_metrics_{self.file_timestamp}.json"
            
            with open(filename, 'w') as f:
                import json
                json.dump(self.agent.training_metrics, f, indent=2)
                
            LOG.debug(f"Saved training metrics to {filename}")
            
        except Exception as e:
            LOG.warning(f"Could not save training metrics: {e}")



    # validating the action and submitting it to the forwarding module
    def migrate_flow(self, src_dpid, flow_id, dst_port):
        """
        Migrate a flow to a new port.
        
        Args:
            src_dpid: Source switch DPID
            flow_id: Flow ID to migrate
            dst_port: Destination port to migrate the flow to
            
        Returns:
            dict: Migration result with success status and message
        """
        LOG.info(f"[Agent2 interface] INFO: Migrating flow {flow_id} from {src_dpid} to port {dst_port}")

        # Check if the flow exists
        #if flow_id not in self.forwarding.mptcp_flows:
        #    return {'success': False, 'message': f'Flow {flow_id} does not exist'}
        LOG.info("[Agent2 interface] INFO: We are skipping checking if it is in mptcp_flows, since we are using cookie_id as flow_id")

        # Perform migration logic here (e.g., update flow rules)
        # This is a placeholder for actual migration logic
        # For example, you might need to remove the old flow and add a new one
        check_result = self.validate_reachable_dst(src_dpid, flow_id, dst_port)
        if not check_result['status']:
            LOG.warning(f"[Agent2 interface] WARNING: Flow {flow_id} cannot be migrated to port {dst_port} from {src_dpid}")
            return check_result
        
        
        # Update the flow implementathion in the forwarding module
        # for later use **************


        LOG.info(f"[Agent2 interface] INFO: Flow {flow_id} migrated successfully to port {dst_port}")
        return check_result
    
    def validate_reachable_dst(self, current_dpid, flow_id, port):
        """
        Validate if the destination DPID is reachable from the source DPID.
        Using SPF we emit the shortest path. from flow_id[path] we remove the switches before the current_dpid.
        Then find the shortest path to the dst_dpid. from the current_dpid to the dst_dpid.

        Args:
            src_dpid: Source switch DPID
            dst_dpid: Destination switch DPID
            
        Returns:
            bool: True if reachable, False otherwise
            path: returns new path to use. 
            path_id: returns path id to use
        """
        result = {'status': True, 'validate': False, 'path': None, 'path_id': None, 'flow_id': None}
        # Get the flow_id from the node_id name
        # flow_id - "flow_switchId-cookieId"
        if not flow_id:
            LOG.warning("[Agent2 interface] WARNING: Flow ID is empty or None")
            return {'status': False, 'validate': False, 'message': 'Flow ID is required'}
        if not current_dpid:
            LOG.warning("[Agent2 interface] WARNING: Current DPID is empty or None")
            return {'status': False, 'validate': False, 'message': 'Current DPID is required'}
        if not port:
            LOG.warning("[Agent2 interface] WARNING: Port is empty or None")
            return {'status': False, 'validate': False, 'message': 'Port is required'}
        cookie_id = int(flow_id.split('-')[-1])  # Extract cookie_id from flow_id
        if not cookie_id:
            LOG.warning("[Agent2 interface] WARNING: Cookie ID is empty or None")
            return {'status': False, 'validate': False, 'message': 'Cookie ID is required'}

        # Get path_key using cookie_id (flow_id is used as cookie_id in this context)
        path_key = self.forwarding.get_path_key_by_flow_id(cookie_id) if self.forwarding else None
        if not path_key:
            LOG.warning(f"[Agent2 interface] WARNING: No path key found for flow {cookie_id}")
            return result
        path = self.topology_manager.get_path_by_key(path_key) if path_key else None
        if not path:
            LOG.warning(f"[Agent2 interface] WARNING: No path found for key {path_key}")
            return result
        dst_dpid = path[-1]  # Last DPID in the path is the destination
        src_dpid = path[0]  # First DPID in the path is the source
        LOG.info(f"[Agent2 interface] INFO: Validating reachability from {current_dpid} to {dst_dpid}")
        # port is new_port_S1-port2
        port = port.split('-')[-1]  # Extract port number from port string
        
        # get the number from the port string
        port = int(port.split('port')[-1])  # Extract port number from port string
        if not isinstance(port, int):
            LOG.warning(f"[Agent2 interface] WARNING: Port {port} is not a valid integer")
            return {'status': False, 'message': 'Port must be an integer'}
        current_dpid = int(current_dpid.split('_')[-1])  # Extract DPID from current_dpid string
        temp_src_dpid = self.topology_manager.get_next_switch_dpid(current_dpid, port)
        if not temp_src_dpid:
            LOG.warning(f"[Agent2 interface] WARNING: Source DPID {current_dpid} not found in topology")
            return result

        # Get the part of the path before src_dpid
        part1_of_path = []
        try:
            src_index = path.index(current_dpid)
            part1_of_path = path[:src_index + 1]  # Include src_dpid in the part1_of_path
        except ValueError:
            LOG.warning(f"[Agent2 interface] WARNING: Source DPID {src_dpid} not found in path {path}")
            return result
        
        # Get shortest path from src_dpid to dst_dpid
        SPF_status = self.topology_manager.validate_get_shortest_path(temp_src_dpid, dst_dpid, part1_of_path) if self.topology_manager else (None, None, [])

        if SPF_status['status'] == 2:
            LOG.warning(f"[Agent2 interface] WARNING: No path found from {src_dpid} to {dst_dpid}")
            return result
        elif SPF_status['status'] == 1:
            LOG.info(f"[Agent2 interface] INFO: New Path found from {src_dpid} to {dst_dpid}: {SPF_status['path']}")
            result['validate'] = True
            result['path_id'] = SPF_status['path_id']
            result['path'] = SPF_status['path']
            result['flow_id'] = flow_id  # Include flow_id in the result
            self.forwarding.update_path_db()
            return result
        elif SPF_status['status'] == 0:
            LOG.info(f"[Agent2 interface] INFO: Path already exists from {src_dpid} to {dst_dpid}: {SPF_status['path']}")
            result['validate'] = True
            result['path_id'] = SPF_status['path_id']
            result['path'] = SPF_status['path']
            result['flow_id'] = flow_id
            return result
        else:
            LOG.warning(f"[Agent2 interface] WARNING: Unexpected SPF status: {SPF_status['status']}")
            return result

    def calculate_metrics(self):
        """
        Calculate and Store current network metrics.
        Metrics include:
            - Maximum Link Utilization (MLU)
            - Standard Deviation of Link Utilization (SDLU)
            - Standard Deviation of Switch MLU (SDSW)
            - Aggregated flow rate
            - number of paths
        file name: metrics_<timestamp>.json with timestamp in mm-dd-yyyy_hh-mm-ss format

        """
        try:
            if not self.network_monitor:
                LOG.warning("NetworkMonitor not available for metrics calculation")
                return
            
            # Get current network state
            '''state = self._get_network_state()
            if not self._validate_network_state(state):
                LOG.warning("Invalid network state for metrics calculation")
                return'''
            
            state = self._calculate_all_switch_mlus()
            #LOG.info(f"[agent2-interface-logging]: Calculated all switch MLUs: {state}")
            
            mlu = state['mlu']
            switch_mlus = state['switch_mlus']
            timestamp = state['timestamp']
            
            # Calculate metrics
            max_mlu = max(switch_mlus.values(), default=0.0)
            #sdlus = [abs(mlu - mlu_val) for mlu_val in switch_mlus.values()]
            #sdlus_stddev = statistics.stdev(sdlus) if len(sdlus) > 1 else 0.0
            sdlus_stddev = statistics.stdev(switch_mlus.values()) if len(switch_mlus) > 1 else 0.0
            link_stddev = statistics.stdev(self.network_monitor.get_all_links_utilization()) if len(switch_mlus) > 1 else 0.0
            aggregated_flow_rate = self.flow_rate_monitor.get_all_mptcp_aggregate_throughput() if self.flow_rate_monitor else {'total_aggregate_rate_mbps': 0.0, 'total_aggregate_rate_bps': 0.0, 'flows': []}
            num_paths = len(self.topology_manager.get_all_paths()) if self.topology_manager else 0
            m = self._calculate_metric_m(self.network_monitor.get_all_links_utilization()) if self.network_monitor else 0.0
            Links_utilizations = self.network_monitor.get_all_links_utilization_dict() if self.network_monitor else {}
            #LOG.info(f"[agent2-interface-logging]: Links Utilizations: {Links_utilizations}")
            # Prepare metrics data
            metrics_data = {
                'timestamp': timestamp,
                'max_mlu': max_mlu,
                'sdlus_stddev': sdlus_stddev,
                'link_stddev': link_stddev,
                'aggregated_flow_rate_mbps': aggregated_flow_rate['total_aggregate_rate_mbps'],
                #'aggregated_flow_rate_bps': aggregated_flow_rate['total_aggregate_rate_bps'],
                'Links_utilizations': Links_utilizations,
                'aggregated_per_flows': aggregated_flow_rate['flows'],
                'num_paths': num_paths,
                'switch_mlus': switch_mlus,
                'm': m,
                'm_reward_history': self.m_reward_history[-1] if self.m_reward_history else 0.0
            }
            
            # Save to file
            
            '''with open(self.metrics_calculation_filename, 'w') as f:
                import json
                json.dump(metrics_data, f, indent=2)'''
            self._append_metrics_record(metrics_data)
            # Update metrics calculation filename
            if time.time() % 120 == 0:  # Update every 2 minutes
                LOG.info(f"[Agent2 Interface][CalculateMetrics] INFO: Metrics calculated and saved to {self.metrics_calculation_filename}")

        except Exception as e:
            LOG.error(f"Error calculating metrics: {e}")
        
    def _periodic_metrics_calculation(self):
        """Periodically calculate and save network metrics"""
        hub.sleep(10)
        while self.enabled:
            try:
                self.calculate_metrics()
            except Exception as e:
                LOG.error(f"Error in periodic metrics calculation: {e}")
            
            hub.sleep(5)

    # ----------------------------------------------------------------------
    # Helper: append ONE metrics dict to a daily JSON-Lines file
    # ----------------------------------------------------------------------
    def _append_metrics_record(self, record: dict) -> None:
        """
        Append `record` as a single line of JSON to the daily metrics file.
        Creates the directory and file if they do not exist.

        File naming:  RL2/data/Metrics/metrics_<mm-dd-yyyy>.json
        Every call adds *one* line => ideal for streaming analytics later.
        """
        try:
            import json, datetime, os
            
            
            path = f"RL2/data/Metrics"
            os.makedirs(path, exist_ok=True)
            file_path = f"{path}/metrics_{self.file_timestamp}.json"

            # 2) Append (not overwrite)
            with open(file_path, 'a') as fp:
                fp.write(json.dumps(record, separators=(',', ':')) + '\n')

            if self.enable_logging:
                LOG.debug(f"[agent2-interface-logging]: Appended metrics to {file_path}")
        except Exception as e:
            LOG.warning(f"[agent2-interface-logging]: Could not append metrics: {e}")


    def _is_active_flows(self) -> bool:
        """Check if there are any active flows in the network."""
        return self.flow_rate_monitor.is_there_active_flows() if self.flow_rate_monitor else False