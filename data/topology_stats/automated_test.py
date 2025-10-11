#!/usr/bin/env python3
"""
Automated Network Testing Script for 3-Layer Topology
Manages MPTCP iperf3 tests and D-ITG TCP traffic generation
"""

import os
import sys
import time
import threading
import subprocess
import signal
import json
import logging
from datetime import datetime
from collections import defaultdict
import psutil
import random

class PortManager:
    """Manages dynamic port allocation for tests"""
    def __init__(self):
        self.iperf_base = 5201
        self.iperf_max = 5299
        self.ditg_base = 8000
        self.ditg_max = 8099
        self.allocated_iperf = set()
        self.allocated_ditg = set()
        self.lock = threading.Lock()
    
    def get_iperf_port(self):
        with self.lock:
            for port in range(self.iperf_base, self.iperf_max):
                if port not in self.allocated_iperf:
                    self.allocated_iperf.add(port)
                    return port
            raise Exception("No available iperf ports")
    
    def get_ditg_port(self):
        with self.lock:
            for port in range(self.ditg_base, self.ditg_max):
                if port not in self.allocated_ditg:
                    self.allocated_ditg.add(port)
                    return port
            raise Exception("No available D-ITG ports")
    
    def release_iperf_port(self, port):
        with self.lock:
            self.allocated_iperf.discard(port)
    
    def release_ditg_port(self, port):
        with self.lock:
            self.allocated_ditg.discard(port)
    
    def reset(self):
        with self.lock:
            self.allocated_iperf.clear()
            self.allocated_ditg.clear()

class ProcessTracker:
    """Tracks all spawned processes for cleanup"""
    def __init__(self):
        self.processes = []
        self.lock = threading.Lock()
    
    def add(self, proc, name):
        with self.lock:
            self.processes.append((proc, name))
    
    def terminate_all(self, logger):
        with self.lock:
            for proc, name in self.processes:
                try:
                    if proc.poll() is None:  # Process still running
                        logger.info(f"Terminating {name} (PID: {proc.pid})")
                        proc.terminate()
                        time.sleep(0.5)
                        if proc.poll() is None:
                            logger.info(f"Force killing {name} (PID: {proc.pid})")
                            proc.kill()
                except Exception as e:
                    logger.error(f"Error terminating {name}: {e}")
            self.processes.clear()
    
    def cleanup_by_name(self, process_names, logger):
        """Kill processes by name (for extra cleanup)"""
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['name'] in process_names:
                    logger.debug(f"Killing lingering process: {proc.info['name']} (PID: {proc.info['pid']})")
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

class AutomatedNetworkTest:
    def __init__(self, net, verify_cleanup=True, test_count=100, use_iperf_fallback=True):
        self.net = net
        self.verify_cleanup = verify_cleanup
        self.test_count = test_count
        self.test_duration = 2100  # 35 minutes
        self.cooldown_time = 120   # 2 minutes between tests
        self.initial_delay = 60    # 1 minute initial delay
        self.use_iperf_fallback = use_iperf_fallback  # Fallback to iperf3 for TCP traffic
        
        # Create test directory
        self.test_dir = self.create_test_directory()
        self.logger = self.setup_logging()
        
        # Initialize managers
        self.port_manager = PortManager()
        self.process_tracker = ProcessTracker()
        
        # Zone definitions
        self.zones = {
            1: {'mptcp': ['h1', 'h2'], 'tcp': ['h11', 'h12']},
            2: {'mptcp': ['h3', 'h4'], 'tcp': ['h13', 'h14']},
            3: {'mptcp': ['h5', 'h6'], 'tcp': ['h15', 'h16']}
        }
        
        # Rotation patterns for zone communication
        self.rotation_patterns = [
            [(1,2), (2,3), (3,1)],  # Pattern 0: 1?2, 2?3, 3?1
            [(1,3), (2,1), (3,2)],  # Pattern 1: 1?3, 2?1, 3?2
            [(2,1), (3,2), (1,3)],  # Pattern 2: 2?1, 3?2, 1?3
            [(2,3), (3,1), (1,2)],  # Pattern 3: 2?3, 3?1, 1?2
            [(3,1), (1,2), (2,3)],  # Pattern 4: 3?1, 1?2, 2?3
            [(3,2), (1,3), (2,1)]   # Pattern 5: 3?2, 1?3, 2?1
        ]
        
        # Thread control
        self.stop_event = threading.Event()
        self.error_event = threading.Event()
        
        # Check if D-ITG is available
        self.ditg_available = self.check_ditg_availability()
    
    def check_ditg_availability(self):
        """Check if D-ITG tools are available"""
        try:
            result = subprocess.run(['which', 'ITGSend'], capture_output=True)
            if result.returncode == 0:
                result = subprocess.run(['which', 'ITGRecv'], capture_output=True)
                if result.returncode == 0:
                    self.logger.info("D-ITG tools found and available")
                    return True
        except:
            pass
        
        self.logger.warning("D-ITG tools not found. Will use iperf3 fallback for TCP traffic.")
        return False
    
    def run_iperf_tcp_server(self, host, port):
        """Start iperf3 server for TCP traffic (fallback option)"""
        cmd = f"iperf3 -s -p {port} --one-off"
        self.logger.debug(f"Starting iperf3 TCP server on {host} port {port}")
        
        host_obj = self.net.get(host)
        proc = host_obj.popen(cmd, shell=True)
        self.process_tracker.add(proc, f"iperf3-tcp-server-{host}")
        return proc
    
    def run_iperf_tcp_client(self, src_host, dst_host, port, test_id, duration):
        """Run iperf3 client for TCP traffic at 3Mbps (fallback option)"""
        output_file = os.path.join(self.test_dir, "ditg", f"{src_host}-{test_id:03d}-{dst_host}-tcp.txt")
        
        # Get destination IP
        dst_obj = self.net.get(dst_host)
        dst_ip = dst_obj.IP()
        
        # Use iperf3 with bandwidth limit of 3Mbps
        cmd = f"iperf3 -c {dst_ip} -p {port} -t {duration} -b 3M -J > {output_file} 2>&1"
        self.logger.debug(f"Starting iperf3 TCP client: {src_host} -> {dst_host} ({dst_ip}:{port}) at 3Mbps")
        
        src_obj = self.net.get(src_host)
        proc = src_obj.popen(cmd, shell=True)
        self.process_tracker.add(proc, f"iperf3-tcp-client-{src_host}")
        return proc
        
    def create_test_directory(self):
        """Create timestamped test directory"""
        timestamp = datetime.now().strftime("%m-%d_%H-%M")
        test_dir = f"test_{timestamp}"
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(os.path.join(test_dir, "iperf"), exist_ok=True)
        os.makedirs(os.path.join(test_dir, "ditg"), exist_ok=True)
        return test_dir
    
    def setup_logging(self):
        """Setup logging configuration"""
        logger = logging.getLogger('NetworkTest')
        logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler(os.path.join(self.test_dir, 'main.log'))
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def get_rotation_config(self, iteration):
        """Get zone communication pattern for current iteration"""
        pattern_idx = iteration % len(self.rotation_patterns)
        return self.rotation_patterns[pattern_idx]
    
    def get_host_pairs(self, rotation_config):
        """Convert zone pairs to host pairs"""
        host_pairs = []
        
        for src_zone, dst_zone in rotation_config:
            # MPTCP pairs: first host sends, second receives
            src_hosts = self.zones[src_zone]['mptcp']
            dst_hosts = self.zones[dst_zone]['mptcp']
            
            # Source zone: first host sends to destination zone's second host
            mptcp_pair = (src_hosts[0], dst_hosts[1])
            
            # TCP pairs follow the same pattern
            tcp_src = self.zones[src_zone]['tcp'][0]  # h11, h13, or h15
            tcp_dst = self.zones[dst_zone]['tcp'][1]  # h12, h14, or h16
            tcp_pair = (tcp_src, tcp_dst)
            
            host_pairs.append({
                'mptcp': mptcp_pair,
                'tcp': tcp_pair,
                'src_zone': src_zone,
                'dst_zone': dst_zone
            })
        
        return host_pairs
    
    def run_iperf_server(self, host, port):
        """Start iperf3 server on specified host"""
        cmd = f"iperf3 -s -p {port}"
        self.logger.debug(f"Starting iperf3 server on {host} port {port}")
        
        host_obj = self.net.get(host)
        proc = host_obj.popen(cmd, shell=True)
        self.process_tracker.add(proc, f"iperf3-server-{host}")
        return proc
    
    def run_iperf_client(self, src_host, dst_host, port, test_id, duration):
        """Run iperf3 client test"""
        output_file = os.path.join(self.test_dir, "iperf", f"{src_host}-{test_id:03d}-{dst_host}.txt")
        
        # Get destination IP
        dst_obj = self.net.get(dst_host)
        dst_ip = dst_obj.IP()
        
        cmd = f"iperf3 -c {dst_ip} -p {port} -t {duration} -J > {output_file} 2>&1"
        self.logger.debug(f"Starting iperf3 client: {src_host} -> {dst_host} ({dst_ip}:{port})")
        
        src_obj = self.net.get(src_host)
        proc = src_obj.popen(cmd, shell=True)
        self.process_tracker.add(proc, f"iperf3-client-{src_host}")
        return proc
    
    def run_ditg_receiver(self, host, port):
        """Start D-ITG receiver"""
        # Create log directory for receiver
        receiver_log = os.path.join(self.test_dir, "ditg", f"receiver_{host}.log")
        cmd = f"ITGRecv -Sp {port} -l {receiver_log}"
        self.logger.debug(f"Starting D-ITG receiver on {host} port {port}")
        
        host_obj = self.net.get(host)
        # Ensure the receiver is ready
        proc = host_obj.popen(cmd, shell=True)
        self.process_tracker.add(proc, f"ditg-receiver-{host}")
        
        # Give receiver time to bind to port
        time.sleep(1)
        return proc
    
    def run_ditg_sender(self, src_host, dst_host, port, test_id, duration):
        """Run D-ITG sender with 3Mbps CBR traffic"""
        output_file = os.path.join(self.test_dir, "ditg", f"{src_host}-{test_id:03d}-{dst_host}.log")
        
        # Get destination IP
        dst_obj = self.net.get(dst_host)
        dst_ip = dst_obj.IP()
        
        # Use direct command line parameters (more reliable than script file)
        cmd = [
            'ITGSend',
            '-a', dst_ip,           # destination address
            '-sp', str(port),       # server port  
            '-C', '3000',           # 3Mbps constant bitrate
            '-t', str(duration*1000), # duration in milliseconds
            '-T', 'TCP',            # TCP protocol
            '-l', output_file       # log file
        ]
        
        cmd_str = ' '.join(cmd)
        self.logger.debug(f"Starting D-ITG sender: {src_host} -> {dst_host} ({dst_ip}:{port}) at 3Mbps")
        self.logger.debug(f"D-ITG command: {cmd_str}")
        
        src_obj = self.net.get(src_host)
        proc = src_obj.popen(cmd_str, shell=True)
        self.process_tracker.add(proc, f"ditg-sender-{src_host}")
        
        # Alternative: If direct command fails, try script file method
        # Uncomment below if needed:
        '''
        # D-ITG script format: destination:protocol:port:bitrate:duration
        script_content = f"{dst_ip}:TCP:{port}:3000:{duration*1000}"
        script_file = f"/tmp/ditg_script_{src_host}_{test_id}.txt"
        
        src_obj = self.net.get(src_host)
        src_obj.cmd(f"echo '{script_content}' > {script_file}")
        
        cmd = f"ITGSend -a {script_file} -l {output_file}"
        proc = src_obj.popen(cmd, shell=True)
        '''
        
        return proc
    
    def mptcp_thread(self, config, test_id):
        """Thread 2: Manage MPTCP iperf3 tests"""
        self.logger.info(f"MPTCP thread started for test {test_id}")
        
        try:
            servers = []
            clients = []
            
            # Start servers first
            for pair_config in config:
                mptcp_src, mptcp_dst = pair_config['mptcp']
                port = self.port_manager.get_iperf_port()
                
                # Start server on destination
                server_proc = self.run_iperf_server(mptcp_dst, port)
                servers.append((server_proc, mptcp_dst, port))
                time.sleep(2)  # Give server time to start
                
                # Store port for client
                pair_config['iperf_port'] = port
            
            # Start clients
            for pair_config in config:
                mptcp_src, mptcp_dst = pair_config['mptcp']
                port = pair_config['iperf_port']
                
                client_proc = self.run_iperf_client(
                    mptcp_src, mptcp_dst, port, test_id, self.test_duration
                )
                clients.append((client_proc, mptcp_src, mptcp_dst))
            
            # Monitor for errors or stop signal
            start_time = time.time()
            while not self.stop_event.is_set() and not self.error_event.is_set():
                # Check if any process died unexpectedly
                for proc, host, _ in servers + clients:
                    if proc.poll() is not None and time.time() - start_time < self.test_duration:
                        self.logger.error(f"Process died unexpectedly on {host}")
                        self.error_event.set()
                        return
                
                time.sleep(5)
            
        except Exception as e:
            self.logger.error(f"Error in MPTCP thread: {e}")
            self.error_event.set()
        finally:
            # Release ports
            for pair_config in config:
                if 'iperf_port' in pair_config:
                    self.port_manager.release_iperf_port(pair_config['iperf_port'])
    
    def ditg_thread(self, config, test_id):
        """Thread 3: Manage D-ITG TCP traffic (or iperf3 fallback)"""
        if self.ditg_available and not self.use_iperf_fallback:
            self.logger.info(f"D-ITG thread started for test {test_id}")
        else:
            self.logger.info(f"TCP traffic thread started for test {test_id} (using iperf3 fallback)")
        
        try:
            receivers = []
            senders = []
            
            # Decide which tool to use
            use_iperf = not self.ditg_available or self.use_iperf_fallback
            
            # Start receivers first
            for pair_config in config:
                tcp_src, tcp_dst = pair_config['tcp']
                
                if use_iperf:
                    port = self.port_manager.get_iperf_port()
                    receiver_proc = self.run_iperf_tcp_server(tcp_dst, port)
                else:
                    port = self.port_manager.get_ditg_port()
                    receiver_proc = self.run_ditg_receiver(tcp_dst, port)
                
                receivers.append((receiver_proc, tcp_dst, port))
                time.sleep(3)  # Give receiver time to start
                
                # Store port for sender
                pair_config['tcp_port'] = port
                pair_config['use_iperf'] = use_iperf
            
            # Start senders
            for pair_config in config:
                tcp_src, tcp_dst = pair_config['tcp']
                port = pair_config['tcp_port']
                
                if pair_config['use_iperf']:
                    sender_proc = self.run_iperf_tcp_client(
                        tcp_src, tcp_dst, port, test_id, self.test_duration
                    )
                else:
                    sender_proc = self.run_ditg_sender(
                        tcp_src, tcp_dst, port, test_id, self.test_duration
                    )
                
                senders.append((sender_proc, tcp_src, tcp_dst))
                time.sleep(1)  # Stagger sender starts
            
            # Monitor for errors or stop signal
            start_time = time.time()
            check_interval = 0
            failed_once = False
            
            while not self.stop_event.is_set() and not self.error_event.is_set():
                # Check if any process died unexpectedly
                for proc, host, _ in receivers + senders:
                    if proc.poll() is not None and time.time() - start_time < self.test_duration:
                        exit_code = proc.poll()
                        
                        # If D-ITG failed and we haven't tried fallback yet
                        if not use_iperf and not failed_once and self.use_iperf_fallback:
                            self.logger.warning(f"D-ITG failed on {host}, switching to iperf3 fallback")
                            failed_once = True
                            self.ditg_available = False  # Disable D-ITG for future tests
                            return  # Let the main thread retry with iperf3
                        
                        self.logger.error(f"TCP traffic process died unexpectedly on {host} with exit code {exit_code}")
                        self.error_event.set()
                        return
                
                # Log status periodically
                if check_interval % 12 == 0:  # Every minute
                    self.logger.debug(f"TCP traffic thread running, elapsed: {int(time.time() - start_time)}s")
                
                check_interval += 1
                time.sleep(5)
            
        except Exception as e:
            self.logger.error(f"Error in TCP traffic thread: {e}")
            self.error_event.set()
        finally:
            # Release ports
            for pair_config in config:
                if 'tcp_port' in pair_config:
                    if pair_config.get('use_iperf', False):
                        self.port_manager.release_iperf_port(pair_config['tcp_port'])
                    else:
                        self.port_manager.release_ditg_port(pair_config['tcp_port'])
    
    def cleanup_all_processes(self):
        """Clean up all test processes"""
        self.logger.info("Cleaning up all processes...")
        
        # Terminate tracked processes
        self.process_tracker.terminate_all(self.logger)
        
        # Extra cleanup if enabled
        if self.verify_cleanup:
            time.sleep(2)
            self.process_tracker.cleanup_by_name(['iperf3', 'ITGSend', 'ITGRecv'], self.logger)
        
        # Reset port manager
        self.port_manager.reset()
    
    def run_single_test(self, test_id):
        """Run a single test iteration"""
        self.logger.info(f"Starting test {test_id} at {datetime.now().isoformat()}")
        
        # Reset events
        self.stop_event.clear()
        self.error_event.clear()
        
        # Get rotation configuration
        rotation_config = self.get_rotation_config(test_id - 1)
        host_pairs = self.get_host_pairs(rotation_config)
        
        self.logger.info(f"Test {test_id} rotation pattern: {rotation_config}")
        
        # Start threads
        mptcp_thread = threading.Thread(
            target=self.mptcp_thread,
            args=(host_pairs, test_id),
            name=f"MPTCP-{test_id}"
        )
        
        ditg_thread = threading.Thread(
            target=self.ditg_thread,
            args=(host_pairs, test_id),
            name=f"DITG-{test_id}"
        )
        
        mptcp_thread.start()
        time.sleep(5)  # Stagger thread starts
        ditg_thread.start()
        
        # Wait for test duration or error
        start_time = time.time()
        while time.time() - start_time < self.test_duration:
            if self.error_event.is_set():
                self.logger.error(f"Error detected in test {test_id}, stopping early")
                break
            time.sleep(10)
        
        # Signal threads to stop
        self.stop_event.set()
        
        # Wait for threads to finish
        mptcp_thread.join(timeout=30)
        ditg_thread.join(timeout=30)
        
        # Clean up processes
        self.cleanup_all_processes()
        
        self.logger.info(f"Completed test {test_id} at {datetime.now().isoformat()}")
        
        # Return success/failure
        return not self.error_event.is_set()
    
    def verify_connectivity(self):
        """Verify basic connectivity between hosts"""
        self.logger.info("Verifying network connectivity...")
        
        # Test connectivity between all zone pairs
        test_pairs = [
            ('h1', 'h5'),   # Zone 1 to Zone 3
            ('h3', 'h1'),   # Zone 2 to Zone 1
            ('h5', 'h3'),   # Zone 3 to Zone 2
            ('h11', 'h15'), # TCP hosts
            ('h13', 'h11'),
            ('h15', 'h13')
        ]
        
        for src, dst in test_pairs:
            src_host = self.net.get(src)
            dst_host = self.net.get(dst)
            dst_ip = dst_host.IP()
            
            # Simple ping test
            result = src_host.cmd(f'ping -c 1 -W 1 {dst_ip}')
            if '1 received' in result:
                self.logger.debug(f"Connectivity OK: {src} -> {dst} ({dst_ip})")
            else:
                self.logger.error(f"Connectivity FAILED: {src} -> {dst} ({dst_ip})")
                return False
        
        self.logger.info("All connectivity tests passed")
        return True
    
    def run_main_loop(self):
        """Main test loop - Thread 1"""
        self.logger.info(f"Starting automated network test with {self.test_count} iterations")
        self.logger.info(f"Test directory: {self.test_dir}")
        self.logger.info(f"Initial delay: {self.initial_delay} seconds")
        
        # Initial delay for network stabilization
        time.sleep(self.initial_delay)
        
        # Verify connectivity before starting tests
        if not self.verify_connectivity():
            self.logger.error("Connectivity verification failed. Aborting tests.")
            return
        
        completed_tests = 0
        test_id = 1
        
        while completed_tests < self.test_count:
            try:
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Main Test Iteration: {test_id}")
                self.logger.info(f"Completed: {completed_tests}/{self.test_count}")
                self.logger.info(f"{'='*60}")
                
                # Run single test
                success = self.run_single_test(test_id)
                
                if success:
                    completed_tests += 1
                    self.logger.info(f"Test {test_id} completed successfully")
                else:
                    self.logger.warning(f"Test {test_id} failed, will retry")
                
                # Cooldown between tests
                if completed_tests < self.test_count:
                    self.logger.info(f"Cooling down for {self.cooldown_time} seconds...")
                    time.sleep(self.cooldown_time)
                
                test_id += 1
                
            except KeyboardInterrupt:
                self.logger.info("Test interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error in main loop: {e}")
                time.sleep(self.cooldown_time)
        
        self.logger.info(f"Test completed. Total successful iterations: {completed_tests}")

def integrate_with_topology(verify_cleanup=True, test_count=100):
    """Integration function to be called from the topology script"""
    from mininet.net import Mininet
    from mininet.node import RemoteController, OVSKernelSwitch
    from mininet.link import TCLink
    
    # Build topology
    topo = ThreeLayerTopo()
    remote_controller = RemoteController('c0', ip='172.29.25.79', port=6653)
    net = Mininet(topo=topo, switch=OVSKernelSwitch, link=TCLink, controller=remote_controller)
    
    # Start network
    net.start()
    
    # Configure MPTCP as in original script
    for i in range(1, 7):
        host = net.get(f'h{i}')
        host.cmd('sysctl -w net.mptcp.enabled=1')
        host.setIP(f'10.0.1.{i}/24', intf=f'h{i}-eth1')
        host.setIP(f'10.0.2.{i}/24', intf=f'h{i}-eth2')
    
    for i in range(11, 17):
        host = net.get(f'h{i}')
        host.cmd('sysctl -w net.mptcp.enabled=0')
        host.setIP(f'10.0.3.{i-10}/24', intf=f'h{i}-eth1')
    
    # Run automated tests
    try:
        tester = AutomatedNetworkTest(net, verify_cleanup=verify_cleanup, test_count=test_count)
        tester.run_main_loop()
    finally:
        net.stop()

if __name__ == '__main__':
    # Standalone execution
    print("This script should be integrated with the topology script.")
    print("Add the following to your topology script:")
    print("\nfrom automated_test import integrate_with_topology")
    print("integrate_with_topology(verify_cleanup=True, test_count=100)")