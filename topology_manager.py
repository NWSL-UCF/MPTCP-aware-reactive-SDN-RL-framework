# topology_manager_v2.py
"""
Enhanced Topology Manager with better organization and Agent2 support
"""

import json
import logging
import time
#import eventlet
from ryu.lib import hub
import networkx as nx
import heapq
from collections import defaultdict, deque
from ryu.base import app_manager
from ryu.controller.handler import set_ev_cls, MAIN_DISPATCHER, CONFIG_DISPATCHER
from ryu.controller import ofp_event
from ryu.lib.packet import packet, ethernet, arp
from ryu.lib import hub  # Add this import for hub.spawn
from ryu.topology import event as topo_event
from ryu.topology.api import get_all_switch, get_all_link, get_switch
from ryu.ofproto import ofproto_v1_3
from events import EventSendPaths
import setting
from multi_head_graph_manager import MultiHeadGraphManager
from graph_stats_monitor import GraphStatsMonitor



LOG = logging.getLogger('ryu.app.topology_manager_v2')


class TopologyManagerV2(app_manager.RyuApp):
    """
    Enhanced Topology Manager with clear separation of concerns
    """
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    
    def __init__(self, *args, **kwargs):
        super(TopologyManagerV2, self).__init__(*args, **kwargs)
        
        # Basic configuration
        self.k = setting.k  # Number of paths for Yen's algorithm
        self.scheduler_interval = setting.scheduler_interval
        self.global_start_time = time.time()
        
        # Core topology data structures
        self.topology = nx.DiGraph()  # Main topology graph
        self.switches = {}  # {dpid: switch_object}
        self.links = {}  # {(src_dpid, dst_dpid): link_info}
        self.hosts = {}  # {host_ip: (dpid, port)}
        
        # Path computation
        self.all_paths = {}  # {path_id: path_details}
        self.all_paths_dpid = {} # {path_id: path}
        self.path_cache = {}  # Cache for computed paths
        
        # Port and link tables
        self.link_port_table = {}  # {(src_dpid, dst_dpid): (src_port, dst_port)}
        self.switch_all_ports_table = {}  # {dpid: set(ports)}
        self.not_use_ports = {}  # {dpid: set(unused_ports)}
        self.active_ports = {}  # {dpid: set(active_ports)}
        
        # Access tables
        self.access_table = {}  # Runtime host location {(dpid, port): (ip, mac)}
        self.access_table_edge = {}  # Persistent edge host locations
        if setting.TOPOLOGY == "BigTopo":
            self.access_table_edge_file = 'data/access_table/access_table_edge_BigTopo.json'
        elif setting.TOPOLOGY == "Toy":
            self.access_table_edge_file = 'data/access_table/access_table_edge.json'
        
        # Agent2 specific graph - separate from main topology
        self.agent2_graph = nx.Graph()  # Undirected graph for agent2
        self.agent2_metrics = {}  # Metrics specific to agent2

        # Initialize multi-head graph manager
        self.multi_head_manager = MultiHeadGraphManager(self)
        
        # Get references to other modules
        self._setup_module_references()

        # Initialize stats monitor
        self.stats_monitor = GraphStatsMonitor()
        self.stats_monitor.topology_manager = self
        self.multi_head_manager.set_stats_monitor(self.stats_monitor)
        self.network_monitor_moule = None
        
        # Topology version tracking
        self.topology_version = 0
        self.last_topology_update = time.time()
        
        # Initialize persistent data
        self._load_access_table_edge()
        
        # Start scheduler
        #self.scheduler_thread = eventlet.spawn_after(0, self._scheduler)
        self.scheduler_thread = hub.spawn(self._scheduler)

        # Add graph property for compatibility
        self.graph = nx.Graph()

        
        LOG.info(f'Enhanced Topology Manager initialized. NAME: {self.name}')
    
    # ==================== Initialization Methods ====================
    
    def _load_access_table_edge(self):
        """Load persistent access table from file"""
        try:
            with open(self.access_table_edge_file, 'r') as f:
                existing_data = json.load(f)
                self.access_table_edge = {
                    tuple(map(int, k.split(':'))): v 
                    for k, v in existing_data.items()
                }
                LOG.info(f"Loaded access_table_edge from {self.access_table_edge_file}")
        except (FileNotFoundError, json.JSONDecodeError):
            with open(self.access_table_edge_file, 'w') as f:
                json.dump({}, f)
                LOG.info(f"Created new file {self.access_table_edge_file}")
    
    def _setup_module_references(self):
        """Setup references to other modules"""
        def setup_refs():
            # Wait a bit for other modules to initialize
            hub.sleep(2)
            
            try:
                # Get flow rate monitor reference
                flow_rate_monitor = app_manager.lookup_service_brick('FlowRateModule')
                network_monitor_moule = app_manager.lookup_service_brick('NetworkMonitor')
                if flow_rate_monitor:
                    self.multi_head_manager.set_flow_rate_monitor(flow_rate_monitor)
                    self.multi_head_manager.set_network_monitor_module(network_monitor_moule)
                    LOG.info("Flow rate monitor reference set in multi-head manager")
                else:
                    LOG.warning("Flow rate monitor not found")
                
                # Get stats monitor reference (existing)
                stats_monitor = app_manager.lookup_service_brick('GraphStatsMonitor')
                if stats_monitor:
                    self.multi_head_manager.set_stats_monitor(stats_monitor)
                    LOG.info("Stats monitor reference set in multi-head manager")
                else:
                    LOG.warning("Stats monitor not found")
            
            except Exception as e:
                LOG.error(f"Error setting up module references: {e}")
        
        # Run setup in background
        hub.spawn(setup_refs)

    # ==================== Event Handlers ====================
    
    @set_ev_cls(topo_event.EventSwitchEnter, MAIN_DISPATCHER)
    @set_ev_cls(topo_event.EventLinkAdd, MAIN_DISPATCHER)
    def handle_topology_add(self, ev):
        """Handle topology addition events"""
        LOG.info("Topology addition event triggered")
        self._update_topology()
        if setting.TOPOLOGY == "BigTopo":
            self._compute_all_paths_big_topo()
        else:
            self._compute_all_paths()
        self.multi_head_manager.update_graph()
        self._share_topology_data()
    
    @set_ev_cls(topo_event.EventLinkDelete, MAIN_DISPATCHER)
    @set_ev_cls(topo_event.EventSwitchLeave, MAIN_DISPATCHER)
    def handle_topology_delete(self, ev):
        """Handle topology deletion events"""
        LOG.info("Topology deletion event triggered")
        self._update_topology()
        self._compute_all_paths()
        self.multi_head_manager.update_graph()
        self._share_topology_data()
    
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Handle switch feature reply"""
        datapath = ev.msg.datapath
        self._install_default_flows(datapath)
    
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """Handle packet-in events for ARP learning"""
        msg = ev.msg
        datapath = msg.datapath
        in_port = msg.match['in_port']
        
        pkt = packet.Packet(msg.data)
        arp_pkt = pkt.get_protocol(arp.arp)
        
        if isinstance(arp_pkt, arp.arp):
            self._learn_host_location(datapath.id, in_port, 
                                    arp_pkt.src_ip, arp_pkt.src_mac)
    
    # ==================== Core Topology Methods ====================
    
    def _update_topology(self):
        """Update the network topology with current switch and link information"""
        LOG.info("Updating network topology")
        
        # Clear current topology
        self.topology.clear()
        self.topology_version += 1
        self.last_topology_update = time.time()
        
        # Add switches
        switches = get_all_switch(self)
        if not switches:
            LOG.warning("No switches found in topology")
            return
            
        for switch in switches:
            dpid = switch.dp.id
            self.topology.add_node(dpid)
            self.switches[dpid] = switch
            
            # Update port information
            self.switch_all_ports_table.setdefault(dpid, set())
            for port in switch.ports:
                self.switch_all_ports_table[dpid].add(port.port_no)
        
        # Add links
        links = get_all_link(self)
        for link in links:
            src_dpid, dst_dpid = link.src.dpid, link.dst.dpid
            src_port, dst_port = link.src.port_no, link.dst.port_no
            
            # Add bidirectional edges
            self.topology.add_edge(src_dpid, dst_dpid, 
                                 src_port=src_port, dst_port=dst_port,
                                 bandwidth=setting.default_bandwidth,
                                 delay=setting.default_delay)
            
            # Update link tables
            self.link_port_table[(src_dpid, dst_dpid)] = (src_port, dst_port)
            self.links[(src_dpid, dst_dpid)] = {
                'src_port': src_port,
                'dst_port': dst_port,
                'bandwidth': setting.default_bandwidth,
                'delay': setting.default_delay,
                'utilization': 0.0
            }
        
        # Update unused ports
        self._update_unused_ports()
        
        LOG.info(f"Topology updated: {len(self.switches)} switches, {len(self.links)} links")

        # Sync with the graph attribute
        self.graph.clear()

        # add switches as nodes
        for dpid in self.switches:
            self.graph.add_node(dpid, type='switch')

        # add links as edges
        for (src_dpid, dst_dpid), link_info in self.links.items():
            self.graph.add_edge(src_dpid, dst_dpid,
                                src_port=link_info['src_port'],
                                dst_port=link_info['dst_port'],
                                bandwidth=link_info['bandwidth'],
                                delay=link_info['delay'],
                                utilization=link_info['utilization'])
    
    def _update_unused_ports(self):
        """Identify ports not used for inter-switch links"""
        for dpid in self.switch_all_ports_table:
            used_ports = set()
            
            # Find all ports used for links
            for (src, dst), (src_port, dst_port) in self.link_port_table.items():
                if src == dpid:
                    used_ports.add(src_port)
                elif dst == dpid:
                    used_ports.add(dst_port)
            
            # Unused ports are potential host ports
            self.not_use_ports[dpid] = self.switch_all_ports_table[dpid] - used_ports
            self.active_ports[dpid] = self.switch_all_ports_table[dpid] - self.not_use_ports[dpid]
    
    # ==================== Path Computation Methods ====================
    
    def _compute_all_paths(self):
        """Compute k-shortest paths between all switch pairs"""
        LOG.info("Computing paths for all switch pairs")
        self.all_paths.clear()
        self.all_paths_dpid.clear()
        self.path_cache.clear()
        
        for src in self.topology.nodes:
            for dst in self.topology.nodes:
                if src != dst:
                    paths = self._yens_algorithm(src, dst, self.k)
                    self._store_paths(src, dst, paths)
        
        LOG.info(f"Computed {len(self.all_paths)} paths")
        LOG.info(f"Computed {len(self.all_paths_dpid)} paths with dpid")

    def _compute_all_paths_big_topo(self):
        """
        Pre-compute the k-shortest paths **only** between the
        six main spine switches (s1 – s6) instead of every node pair.

        Results go into:
            • self.all_paths[(src, dst)]           → list[list[str]]
            • self.all_paths_dpid[(src, dst)]      → list[list[int]]
            • self.path_cache[(src, dst, k_idx)]   → list[str]  (if you cache per-k path)
        """
        LOG.info("Computing paths for switches s1-s6 only")

        # ------------------------------------------------------------------
        # 1) Reset caches
        # ------------------------------------------------------------------
        self.all_paths.clear()
        self.all_paths_dpid.clear()
        self.path_cache.clear()

        # ------------------------------------------------------------------
        # 2) Define the subset we care about
        # ------------------------------------------------------------------
        core_switches = [1, 2, 3, 4, 5, 6]


        # Make sure they actually exist in the topology graph
        
        if len(self.topology.nodes) < 18:
            LOG.warning(
                f"Some requested switches are missing from the topology: "
                f"{set(['1','2','3','4','5','6']) - set(core_switches)}"
            )
            LOG.info(f"[topology_manager] INFO: nodes in topology: {self.topology.nodes}")
            LOG.info(f"[topology_manager] INFO: node types: {type(self.topology.nodes)}")
        else:
            LOG.info(f"[topology_manager] INFO: All core switches found in topology: {core_switches}")
            core_switches = [sw for sw in core_switches if sw in self.topology.nodes]
            LOG.info(f"[topology_manager] INFO: Filtered core switches: {core_switches}")

            # ------------------------------------------------------------------
            # 3) Compute paths for each ordered pair  (s_i ≠ s_j)
            # ------------------------------------------------------------------
            for src in core_switches:
                for dst in core_switches:
                    if src == dst:
                        continue

                    paths = self._yens_algorithm(src, dst, self.k)
                    self._store_paths(src, dst, paths)   # <- your existing helper

            LOG.info(
                f"Computed paths for {len(core_switches)}×{len(core_switches)-1} "
                f"ordered switch pairs → {len(self.all_paths)} path sets "
                f"({sum(len(p) for p in self.all_paths.values())} total paths)"
            )


    def _yens_algorithm(self, source, target, k):
        """Yen's k-shortest paths algorithm"""
        if (source, target) in self.path_cache:
            return self.path_cache[(source, target)]
        
        try:
            # Find shortest path first
            A = [nx.shortest_path(self.topology, source, target, weight='weight')]
            B = []
            
            for i in range(1, k):
                for j in range(len(A[-1]) - 1):
                    spur_node = A[-1][j]
                    root_path = A[-1][:j + 1]
                    
                    # Temporarily remove edges
                    removed_edges = []
                    for path in A:
                        if len(path) > j and root_path == path[:j + 1]:
                            edge = (path[j], path[j + 1])
                            if self.topology.has_edge(*edge):
                                edge_data = self.topology[edge[0]][edge[1]]
                                self.topology.remove_edge(*edge)
                                removed_edges.append((edge, edge_data))
                    
                    # Find spur path
                    try:
                        spur_path = nx.shortest_path(self.topology, spur_node, target, weight='weight')
                        total_path = root_path[:-1] + spur_path
                        
                        if self._is_valid_path(total_path) and total_path not in B:
                            B.append(total_path)
                    except nx.NetworkXNoPath:
                        pass
                    
                    # Restore edges
                    for edge, data in removed_edges:
                        self.topology.add_edge(edge[0], edge[1], **data)
                
                if not B:
                    break
                    
                # Sort by path length and add shortest to A
                B.sort(key=len)
                A.append(B.pop(0))
            
            self.path_cache[(source, target)] = A
            return A
            
        except nx.NetworkXNoPath:
            LOG.warning(f"No path found between {source} and {target}")
            return []
    
    def _is_valid_path(self, path):
        """Check if path contains no loops"""
        return len(path) == len(set(path))
    
    def _store_paths(self, src, dst, paths):
        """Store computed paths with detailed information"""
        for idx, path in enumerate(paths):
            path_id = f"{src}-{dst}-path{idx + 1}"
            
            # Create detailed path with link information
            detailed_path = []
            for i in range(len(path) - 1):
                node = path[i]
                next_node = path[i + 1]
                
                if self.topology.has_edge(node, next_node):
                    link_data = self.topology[node][next_node]
                    detailed_path.append([node, link_data])
                else:
                    detailed_path.append([node, None])
            
            # Add last node
            if path:
                detailed_path.append([path[-1], None])
            
            self.all_paths[path_id] = detailed_path
            self.all_paths_dpid[path_id] = path
    def _store_path(self, src, dst, path):
        """Store computed paths with detailed information"""
        
        # Check if path already exists
        for path_id, existing_path in self.all_paths_dpid.items():
            if existing_path == path:
                LOG.warning(f"[topology_manager] WARN: Path already exists for {src} to {dst} with ID {path_id}")
                return path_id

        path_id = f"{src}-{dst}-path"
        for path_id, existing_path in self.all_paths_dpid.items():
            if path_id.startswith(f"{src}-{dst}-path"):
                # Find the maximum path ID for this src-dst pair
                LOG.info(f"[topology_manager] INFO: Found existing path ID {path_id} for {src} to {dst}")
                max_id = int(path_id.split('-')[-1].replace('path', ''))
        
        LOG.info(f"[topology_manager] INFO: Storing path {src} to {dst} with ID {max_id + 1}")
        path_id = f"{src}-{dst}-path{max_id + 1}"
        
        # Create detailed path with link information
        detailed_path = []
        for i in range(len(path) - 1):
            node = path[i]
            next_node = path[i + 1]
            
            if self.topology.has_edge(node, next_node):
                link_data = self.topology[node][next_node]
                detailed_path.append([node, link_data])
            else:
                detailed_path.append([node, None])
        
        # Add last node
        if path:
            detailed_path.append([path[-1], None])
        
        self.all_paths[path_id] = detailed_path
        self.all_paths_dpid[path_id] = path
        LOG.info(f"[topology_manager] INFO: pathID : {path_id} All_path_dpid {self.all_paths_dpid}")
        return path_id

    def get_all_paths(self):
        """Get all computed paths"""
        return self.all_paths.copy()

    # ==================== Agent2 Graph Methods ====================
    
    def _update_agent2_graph(self):
        """Update the separate graph for Agent2 with specific metrics"""
        LOG.info("Updating Agent2 graph")
        
        # Clear and rebuild agent2 graph
        self.agent2_graph.clear()
        
        # Copy topology structure (undirected for agent2)
        for node in self.topology.nodes:
            self.agent2_graph.add_node(node, 
                                     type='switch',
                                     load=0.0,
                                     flows=0)
        
        # Add edges with agent2-specific attributes
        for src, dst, data in self.topology.edges(data=True):
            # Calculate agent2 metrics
            bandwidth = data.get('bandwidth', setting.default_bandwidth)
            delay = data.get('delay', setting.default_delay)
            utilization = self.links.get((src, dst), {}).get('utilization', 0.0)
            
            # Agent2 specific weight calculation
            agent2_weight = self._calculate_agent2_weight(bandwidth, delay, utilization)
            
            self.agent2_graph.add_edge(src, dst,
                                     bandwidth=bandwidth,
                                     delay=delay,
                                     utilization=utilization,
                                     weight=agent2_weight)
        
        # Update agent2 metrics
        self._update_agent2_metrics()
    
    def _calculate_agent2_weight(self, bandwidth, delay, utilization):
        """Calculate edge weight for agent2 based on multiple metrics"""
        # Example weight function - customize based on agent2 needs
        # Lower weight = better path
        weight = (delay * 0.4 +                    # Delay component
                 (1 - bandwidth/1000) * 0.3 +     # Bandwidth component  
                 utilization * 0.3)                # Utilization component
        return max(0.1, weight)  # Ensure positive weight
    
    def _update_agent2_metrics(self):
        """Update metrics specific to agent2"""
        # Handle empty graph case
        if not self.agent2_graph:
            self.agent2_metrics = {
                'graph_density': 0.0,
                'average_degree': 0.0,
                'is_connected': False,
                'diameter': -1,
                'num_paths': 0,
                'topology_version': self.topology_version
            }
            return
        
        degrees = dict(self.agent2_graph.degree()).values()
        self.agent2_metrics = {
            'graph_density': nx.density(self.agent2_graph),
            'average_degree': sum(degrees) / len(degrees) if degrees else 0,
            'is_connected': nx.is_connected(self.agent2_graph),
            'diameter': nx.diameter(self.agent2_graph) if nx.is_connected(self.agent2_graph) else -1,
            'num_paths': len(self.all_paths),
            'topology_version': self.topology_version
        }
    
    # ==================== Agent2 Query Interface ====================
    
    def get_agent2_graph(self):
        """Get the current agent2 graph"""
        return self.agent2_graph.copy()
    
    def get_agent2_path(self, src, dst, criteria='weight'):
        """Get optimal path for agent2 based on specified criteria"""
        try:
            if criteria == 'weight':
                path = nx.shortest_path(self.agent2_graph, src, dst, weight='weight')
            elif criteria == 'hops':
                path = nx.shortest_path(self.agent2_graph, src, dst)
            elif criteria == 'bandwidth':
                # Invert bandwidth for shortest path (higher bandwidth = lower weight)
                temp_graph = self.agent2_graph.copy()
                for u, v, data in temp_graph.edges(data=True):
                    temp_graph[u][v]['inv_bandwidth'] = 1.0 / max(data['bandwidth'], 1)
                path = nx.shortest_path(temp_graph, src, dst, weight='inv_bandwidth')
            else:
                path = nx.shortest_path(self.agent2_graph, src, dst)
            
            return {
                'path': path,
                'criteria': criteria,
                'metrics': self._get_path_metrics(path)
            }
        except nx.NetworkXNoPath:
            return None
    
    def get_agent2_alternative_paths(self, src, dst, k=3):
        """Get k alternative paths for agent2"""
        paths = []
        
        # Use different criteria to find diverse paths
        for criteria in ['weight', 'hops', 'bandwidth']:
            path_info = self.get_agent2_path(src, dst, criteria)
            if path_info and path_info['path'] not in [p['path'] for p in paths]:
                paths.append(path_info)
        
        # If we need more paths, use k-shortest
        if len(paths) < k:
            try:
                k_paths = list(nx.shortest_simple_paths(self.agent2_graph, src, dst, weight='weight'))
                for path in k_paths[:k]:
                    path_info = {
                        'path': path,
                        'criteria': 'k-shortest',
                        'metrics': self._get_path_metrics(path)
                    }
                    if path_info['path'] not in [p['path'] for p in paths]:
                        paths.append(path_info)
                        if len(paths) >= k:
                            break
            except:
                pass
        
        return paths[:k]
    
    def _get_path_metrics(self, path):
        """Calculate metrics for a given path"""
        if len(path) < 2:
            return {}
        
        total_delay = 0
        min_bandwidth = float('inf')
        max_utilization = 0
        
        for i in range(len(path) - 1):
            if self.agent2_graph.has_edge(path[i], path[i+1]):
                edge_data = self.agent2_graph[path[i]][path[i+1]]
                total_delay += edge_data.get('delay', 0)
                min_bandwidth = min(min_bandwidth, edge_data.get('bandwidth', 0))
                max_utilization = max(max_utilization, edge_data.get('utilization', 0))
        
        return {
            'total_delay': total_delay,
            'min_bandwidth': min_bandwidth,
            'max_utilization': max_utilization,
            'hop_count': len(path) - 1
        }
    
    def get_all_active_ports_table(self, dpid):
        """Get all active ports for a switch"""
        return self.active_ports.get(dpid, set()) if dpid in self.active_ports else set()
    
    # ==================== Host Management Methods ====================
    
    def _learn_host_location(self, dpid, port, ip, mac):
        """Learn and store host location"""
        key = (dpid, port)
        
        # Update runtime access table
        if port in self.not_use_ports.get(dpid, set()):
            self.access_table[key] = (ip, mac)
            self.hosts[ip] = key
            
            # Update persistent edge table
            self._update_edge_table(dpid, port, ip, mac)
    
    def _update_edge_table(self, dpid, port, ip, mac):
        """Update persistent edge host table"""
        key = (dpid, port)
        key_str = f"{dpid}:{port}"
        
        # Check if already stored
        for stored_ip, _ in self.access_table_edge.values():
            if stored_ip == ip:
                return
        
        # Update tables
        self.access_table_edge[key] = (ip, mac)
        
        # Save to file
        try:
            with open(self.access_table_edge_file, 'r') as f:
                existing_data = json.load(f)
        except:
            existing_data = {}
        
        existing_data[key_str] = (ip, mac)
        
        with open(self.access_table_edge_file, 'w') as f:
            json.dump(existing_data, f)
            LOG.info(f"Updated edge table with host {ip} at {dpid}:{port}")
    
    def get_host_location(self, host_ip):
        """Get switch and port for a host IP"""
        if host_ip in ["0.0.0.0", "255.255.255.255"]:
            return None
            
        # Check runtime table first
        if host_ip in self.hosts:
            dpid, port = self.hosts[host_ip]
            return (dpid, port)
        
        # Check persistent table
        for (dpid, port), (ip, mac) in self.access_table_edge.items():
            if ip == host_ip:
                return (dpid, port)
        
        return None
    
    # ==================== Utility Methods ====================
    
    def _install_default_flows(self, datapath):
        """Install default flow rules"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        # ARP flow
        match = parser.OFPMatch(eth_type=setting.ETH_TYPE_ARP)
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, 
                                        ofproto.OFPCML_NO_BUFFER)]
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=setting.PRIORITY_ARP,
                               match=match, instructions=inst)
        datapath.send_msg(mod)
        
        # IPv4 flow
        match_ip = parser.OFPMatch(eth_type=0x0800)
        mod_ip = parser.OFPFlowMod(datapath=datapath, priority=setting.PRIORITY_IPV4,
                                   match=match_ip, instructions=inst)
        datapath.send_msg(mod_ip)
        
        LOG.info(f'Installed default flows on switch {datapath.id}')
    
    def _share_topology_data(self):
        """Share topology data with other modules"""
        try:
            # Update multi-head graph
            if self.multi_head_manager:
                self.multi_head_manager.update_graph()
            
            # Start stats monitoring if not already active
            self.start_stats_monitoring()
            
            LOG.debug("Topology data shared with other modules")
        except Exception as e:
            LOG.error(f"Error sharing topology data: {e}")

    def start_stats_monitoring(self):
        """Start real-time statistics monitoring"""
        if self.stats_monitor:
            self.stats_monitor.start_monitoring()
            LOG.info("Started graph statistics monitoring")

    def stop_stats_monitoring(self):
        """Stop real-time statistics monitoring"""
        if self.stats_monitor:
            self.stats_monitor.stop_monitoring()
            LOG.info("Stopped graph statistics monitoring")

    def _scheduler(self):
        """Periodic topology update scheduler"""
        while True:
            LOG.info("Scheduler: Updating topology")
            self._update_topology()
            #self._compute_all_paths()
            #self.multi_head_manager.update_graph()
            #self._share_topology_data()
            #eventlet.sleep(self.scheduler_interval)
            hub.sleep(self.scheduler_interval)
            #hub.sleep(self.scheduler_interval)
    
    # ==================== API Methods for Dashboard ====================
    
    def get_topology_summary(self):
        """Get topology summary for API/Dashboard"""
        return {
            'num_switches': len(self.switches),
            'num_links': len(self.links),
            'num_hosts': len(self.hosts),
            'num_paths': len(self.all_paths),
            'topology_version': self.topology_version,
            'last_update': self.last_topology_update,
            'is_connected': nx.is_weakly_connected(self.topology),
            'agent2_metrics': self.agent2_metrics
        }
    
    def get_topology_details(self):
        """Get detailed topology information"""
        return {
            'switches': list(self.switches.keys()),
            'links': [{'src': src, 'dst': dst, 'info': info} 
                     for (src, dst), info in self.links.items()],
            'hosts': {ip: {'dpid': loc[0], 'port': loc[1]} 
                     for ip, loc in self.hosts.items()},
            'paths_sample': dict(list(self.all_paths.items())[:10])  # First 10 paths
        }
    
    def get_switch_info(self, dpid):
        """Get information about a specific switch"""
        if dpid not in self.switches:
            return None
            
        neighbors = list(self.topology.neighbors(dpid))
        ports = self.switch_all_ports_table.get(dpid, set())
        unused_ports = self.not_use_ports.get(dpid, set())
        
        return {
            'dpid': dpid,
            'neighbors': neighbors,
            'total_ports': len(ports),
            'unused_ports': list(unused_ports),
            'connected_hosts': [{'ip': ip, 'port': port} 
                               for (sw, port), (ip, _) in self.access_table.items() 
                               if sw == dpid]
        }
    
    def get_path_info(self, src, dst):
        """Get path information between two points"""
        paths = []
        for path_id, path_data in self.all_paths.items():
            if path_id.startswith(f"{src}-{dst}-"):
                simple_path = [node[0] for node in path_data]
                paths.append({
                    'path_id': path_id,
                    'nodes': simple_path,
                    'hop_count': len(simple_path) - 1
                })
        return paths
    
    def get_path_without_info(self, src, dst):
        """Get simple path without detailed link information"""
        if (src, dst) in self.all_paths_dpid:
            return self.all_paths_dpid[(src, dst)]
        if (src, dst) in self.path_cache:
            LOG.debug(f"Using cached path for {src} to {dst}")
            return self.path_cache[(src, dst)]
        
        try:
            path = nx.shortest_path(self.topology, src, dst, weight='weight')
            return path
        except nx.NetworkXNoPath:
            LOG.warning(f"No path found between {src} and {dst}")
            return []
    
    def _cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'scheduler_thread') and self.scheduler_thread:
            #eventlet.kill(self.scheduler_thread)
            hub.kill(self.scheduler_thread)
            self.scheduler_thread = None
        
        LOG.info("Topology Manager cleaned up")


    # ==================== OLD functions ==================== #

    def get_host_ip_location(self, host_ip):
        """Get the switch location (dpid, in_port) of a host by its IP"""
        if host_ip == "0.0.0.0" or host_ip == "255.255.255.255":
            return None
        
        # Check runtime access table first
        for (dpid, in_port), (src_ip, src_mac) in self.access_table.items():
            if src_ip == host_ip:
                return (dpid, in_port)
        
        # Check persistent access table
        for (dpid, in_port), (src_ip, src_mac) in self.access_table_edge.items():
            if src_ip == host_ip:
                return (dpid, in_port)

        LOG.info(f"Host IP {host_ip} location not found")
        return None

    def store_access_table(self, dpid, in_port, src_ip, src_mac):
        """Store host access information in both runtime and persistent tables"""
        key = (dpid, in_port)
        key_str = f"{dpid}:{in_port}"

        # Load existing data from the JSON file
        try:
            with open(self.access_table_edge_file, 'r') as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = {}

        # Convert existing_data keys back to tuples
        self.access_table_edge = {
            tuple(map(int, k.split(':'))): v for k, v in existing_data.items()
        }

        # Check if the host IP is already stored
        if any(src_ip == stored_src_ip for stored_src_ip, _ in self.access_table_edge.values()):
            pass
        else:
            # Update access_table_edge
            self.access_table_edge[key] = (src_ip, src_mac)
            existing_data[key_str] = (src_ip, src_mac)

            # Write the updated data back to the JSON file
            with open(self.access_table_edge_file, 'w') as f:
                json.dump(existing_data, f)
                LOG.info(f"Updated {self.access_table_edge_file} with new access_table_edge")

        if in_port in self.not_use_ports.get(dpid, set()):
            if (dpid, in_port) in self.access_table:
                if self.access_table[(dpid, in_port)] == (src_ip, src_mac):
                    return
                else:
                    self.access_table[(dpid, in_port)] = (src_ip, src_mac)
                    return
            else:
                self.access_table.setdefault((dpid, in_port), None)
                self.access_table[(dpid, in_port)] = (src_ip, src_mac)
                return

    def build_topology_between_switches(self, bw=0, delay=0):
        """Build a network topology graph using the switches and links information"""
        _graph = nx.Graph()
        for link in get_all_link(self):
            src_dpid = link.src.dpid
            dst_dpid = link.dst.dpid
            src_port = link.src.port_no
            dst_port = link.dst.port_no
            _graph.add_edge(src_dpid, dst_dpid, src_port=src_port, dst_port=dst_port, bw=bw, delay=delay)

        if _graph.edges == self.graph.edges:
            return 
        else:
            self.graph = _graph

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _paclet_in_arp_handler(self, ev):
        """Handle PacketIn events and extract host information from ARP packets"""
        msg = ev.msg
        datapath = msg.datapath
        in_port = msg.match['in_port']
        pkt = packet.Packet(msg.data)
        arp_pkt = pkt.get_protocol(arp.arp)

        if isinstance(arp_pkt, arp.arp):
            arp_src_ip = arp_pkt.src_ip
            src_mac = arp_pkt.src_mac
            self.store_access_table(datapath.id, in_port, arp_src_ip, src_mac)


    # ==================== Rest API ====================

    def get_topology_summary(self):
        """Get topology summary statistics"""
        return {
            'switch_count': len(self.switches),
            'link_count': len(self.links),
            'host_count': len(self.hosts),
            'path_count': len(self.all_paths),
            'topology_version': self.topology_version
        }
    
    def get_topology_details(self):
        """Get detailed topology information for visualization"""
        nodes = []
        links = []
        
        # Add switch nodes (circular - standard for network devices)
        for dpid in self.switches:
            nodes.append({
                'id': str(dpid),
                'type': 'switch',
                'dpid': dpid,
                'label': f'S{dpid}',
                'shape': 'circle',
                'color': '#4CAF50',
                'size': 30,
                'font': {'color': 'white', 'size': 12}
            })
        
        # Add host nodes (rectangular - standard for end devices)
        for ip, (dpid, port) in self.hosts.items():
            nodes.append({
                'id': ip,
                'type': 'host',
                'ip': ip,
                'label': ip,
                'connected_to': str(dpid),
                'port': port,
                'shape': 'box',  # rectangular shape for hosts
                'color': '#2196F3',
                'size': 20,
                'font': {'color': 'white', 'size': 10},
                'widthConstraint': {'minimum': 80, 'maximum': 120}  # for IP addresses
            })
        
        # Add links between switches
        for (src_dpid, dst_dpid), link_info in self.links.items():
            links.append({
                'id': f'{src_dpid}-{dst_dpid}',
                'from': str(src_dpid),  # vis.js uses 'from'/'to'
                'to': str(dst_dpid),
                'source': str(src_dpid),  # D3.js uses 'source'/'target'
                'target': str(dst_dpid),
                'src_port': link_info.get('src_port'),
                'dst_port': link_info.get('dst_port'),
                'bandwidth': link_info.get('bandwidth', 0),
                'delay': link_info.get('delay', 0),
                'utilization': link_info.get('utilization', 0),
                'color': '#999999',
                'width': 2,
                'type': 'switch_link'
            })
        
        # Add links from hosts to switches
        for ip, (dpid, port) in self.hosts.items():
            links.append({
                'id': f'{ip}-{dpid}',
                'from': ip,
                'to': str(dpid),
                'source': ip,
                'target': str(dpid),
                'type': 'host_link',
                'port': port,
                'color': '#2196F3',
                'width': 1,
                'dashes': True  # dashed line for host connections
            })
        
        return {
            'nodes': nodes,
            'links': links,
            'options': {
                'nodes': {
                    'borderWidth': 2,
                    'shadow': True
                },
                'edges': {
                    'shadow': True,
                    'smooth': {'type': 'continuous'}
                },
                'physics': {
                    'enabled': True,
                    'stabilization': {'iterations': 100}
                }
            }
        }
    
    def get_switch_info(self, dpid):
        """Get detailed information about a specific switch"""
        if dpid not in self.switches:
            return None
        
        switch_info = self.switches[dpid].copy()
        switch_info.update({
            'dpid': dpid,
            'ports': list(self.switch_all_ports_table.get(dpid, [])),
            'unused_ports': list(self.not_use_ports.get(dpid, [])),
            'connected_hosts': [ip for ip, (host_dpid, port) in self.hosts.items() if host_dpid == dpid],
            'neighbors': [dst for (src, dst) in self.links.keys() if src == dpid]
        })
        
        return switch_info
    
    def get_path_info(self, src, dst):
        """Get path information between source and destination"""
        paths = []
        path_prefix = f"{src}-{dst}-path"
        
        for path_id, path_details in self.all_paths.items():
            if path_id.startswith(path_prefix):
                paths.append({
                    'path_id': path_id,
                    'path': path_details,
                    'hop_count': len(path_details) - 1 if path_details else 0
                })
        
        return paths
    

    # Add new methods for multi-head graph access:
    def get_multi_head_graph_data(self):
        """Get multi-head graph data for GNN processing"""
        return self.multi_head_manager.get_multi_head_data()

    def get_head_data(self, head_id):
        """Get data for a specific head"""
        return self.multi_head_manager.get_head_data(head_id)

    def get_multi_head_statistics(self):
        """Get multi-head graph statistics"""
        return self.multi_head_manager.get_graph_statistics()
    
    def setup_module_references(self):
        """Setup references to other modules"""
        
        def setup_refs():
            try:
                # Get multi-head manager
                
                
                # Get stats monitor
                self.network_monitor_module = app_manager.lookup_service_brick('NetworkMonitor')
            except Exception as e:
                LOG.error(f"Error setting up module references: {e}")


        # Run setup in a separate thread to avoid blocking
        #self.setup_thread = eventlet.spawn(setup_refs)
        self.setup_thread = hub.spawn(setup_refs)

    def get_link_port_table(self):
        """Get the link port table"""
        return self.link_port_table.copy()
    
    def get_next_switch_dpid(self, dpid, port):
        """Get the next switch DPID based on current DPID and port"""
        LOG.info(f"[Topology Manager] Getting next switch for DPID {dpid} on port {port}")
        LOG.info(f"[Topology Manager] Link port table: {self.link_port_table}")
        for (src, dst), (src_port, dst_port) in self.link_port_table.items():
            if src == dpid and src_port == port:
                return dst
        LOG.warning(f"[Topology Manager] WARNING: No next switch found for DPID {dpid} on port {port}")
        return None
    
    def validate_get_shortest_path(self, src, dst, remove_dpids):
        """Validate and get the shortest path between two switches, excluding a specific DPID
            Args:
                src_dpid: Source switch DPID
                dst_dpid: Destination switch DPID
                
            Returns:
                int: 0,1, or 2  depends on the path found
                If path found: 
                    0: path found, new path need to update path_db
                    1: path found, already exists in path_db
                    2: unreachable, no path found
                path_id: returns path id to use. 
                path: returns path to use
        """
        if src in remove_dpids:
            LOG.warning(f"[Topology Manager] WARN: Source DPID {src} is in the list of DPIDs to remove, cannot compute path")
            return {
                'status': 2,
                'path_id': None,
                'path': []
            }
        result = {
            'status': None,
            'path_id': None,
            'path': []
        }
        if src == dst:
            LOG.info(f"[Topology Manager] INFO: Source and destination are the same: {src}")
            path = remove_dpids + [src]
            LOG.info(f"[Topology Manager] INFO: Path is {path}")
            
            #check if path already exists
            check, path_id = self.check_path_exists(path)
            if check:
                result['status'] = 0
                result['path_id'] = path_id
                result['path'] = path
                return result
            else:
                result['status'] = 1
                result['path_id'] = path_id
                result['path'] = path
                return result

        graph = self.topology.copy()
        # Ensure the specified DPID is valid
        # Remove the specified DPID from the graph
        for remove_dpid in remove_dpids:
            if remove_dpid not in self.topology:
                LOG.warning(f"[Topology Manager] WARN: DPID {remove_dpid} not found in topology, skipping removal")
                continue
            LOG.info(f"[Topology Manager] INFO: Removing DPID {remove_dpid} from graph")
            # Remove the node and its edges
            graph.remove_node(remove_dpid)
        LOG.info(f"[Topology Manager] INFO: Graph after removing {remove_dpids}: {graph.nodes()}")

        path = [nx.shortest_path(graph, source=src, target=dst, weight='weight', method='dijkstra')]
        if not path:
            LOG.warning(f"[Topology Manager] WARN: No path found between {src} and {dst} excluding {remove_dpids}")
            result['status'] = 2
            result['path_id'] = None
            result['path'] = []
            return result
        # Check if the path already exists in the stored paths
        LOG.info(f"[Topology Manager] INFO: Path found: {path[0]}")
        LOG.info(f"[Topology Manager] INFO: Removing DPID {remove_dpids} from path")
        complete_path =  remove_dpids + path[0]
        LOG.info(f"[Topology Manager] INFO: Complete path to check: {complete_path}")
        src = complete_path[0]
        dst = complete_path[-1]
        check, path_id = self.check_path_exists(complete_path)
        LOG.info(f"[Topology Manager] INFO: Path check result: {check}, Path ID: {path_id}")

        if check:
            LOG.info(f"[Topology Manager] INFO: Path already exists in path_db: {path_id}")
            result['status'] = 0
            result['path_id'] = path_id
            result['path'] = complete_path
            return result
        else:
            LOG.info(f"[Topology Manager] INFO: Path does not exist, storing new path")
            # Store the new path
            path_id = self._store_path(src, dst, complete_path)
            result['status'] = 1
            result['path_id'] = path_id
            result['path'] = complete_path
            return result

        # Check if path already exists
        path_id = f"{src}-{dst}-path1"
        if path_id in self.all_paths:
            return 1, path_id, self.all_paths[path_id]
        
        # Compute new path excluding the specified DPID
        try:
            temp_graph = self.topology.copy()
            temp_graph.remove_node(remove_dpid)
            path = nx.shortest_path(temp_graph, source=src, target=dst, weight='weight')
            
            # Store the new path
            self._store_paths(src, dst, [path])
            return 0, path_id, path
        
        except nx.NetworkXNoPath:
            LOG.warning(f"No path found between {src} and {dst} excluding {remove_dpid}")
            return 2, None, []
        

    def check_path_exists(self, path):
        """Check if a path exists between two switches
        Args:
            path (list): List of switch DPIDs representing the path
        Returns:
            bool: True if path exists, False otherwise
            path_id (str): Path ID if exists, None otherwise
        """

        for path_id, path_data in self.all_paths.items():
            if len(path_data) == len(path) and all(node[0] == dpid for node, dpid in zip(path_data, path)):
                LOG.info(f"[Topology Manager] INFO: Path {path_id} exists for path: {path}")
                return True, path_id

        LOG.info(f"[Topology Manager] INFO: No existing path found for: {path}")
        # adding path and getting new path_id
        src = path[0]
        dst = path[-1]
        
        path_id = self._store_path(src, dst, path)
        LOG.info(f"[Topology Manager] INFO: New path {path_id} added for path: {path}")

        return False, path_id

    def get_path_by_key(self, key):
        """Get path by key from all_paths_dpid"""
        if key in self.all_paths_dpid:
            return self.all_paths_dpid[key]
        else:
            LOG.warning(f"[Topology Manager] WARNING: Path with key {key} not found")
            return None
        
    def get_all_active_switches(self):
        """Get all active switches in the topology"""
        return list(self.switches.keys())