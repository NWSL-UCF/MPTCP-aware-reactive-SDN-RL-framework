#!/usr/bin/env python3
"""
Test script for BigTopo.py
Runs 10 different traffic patterns, each for 1000 seconds
Pattern: Z1->Z2->Z3 where odd hosts send and even hosts receive
"""

import time
import os
import sys
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.log import setLogLevel, info
from mininet.link import TCLink
import json
from pathlib import Path
from BigTopo import ThreeLayerTopo
from typing import List, Dict, Tuple, Union  # Add this import

def load_traffic_patterns(json_file: str = 'test_pattern.json') -> List[Dict[str, Union[str, List[Tuple[str, str, int]]]]]:
    """Load traffic patterns from JSON file"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Convert list format to tuple format for compatibility
        patterns = []
        for pattern in data:
            flows = [(flow[0], flow[1], flow[2]) for flow in pattern['flows']]
            patterns.append({
                'name': pattern['name'],
                'flows': flows
            })
        
        info(f'*** Loaded {len(patterns)} traffic patterns from {json_file}\n')
        return patterns
    
    except FileNotFoundError:
        info(f'*** Warning: {json_file} not found, using default patterns\n')
        return get_default_patterns()
    except json.JSONDecodeError as e:
        info(f'*** Error parsing {json_file}: {e}\n')
        return get_default_patterns()

def get_default_patterns() -> List[Dict[str, Union[str, List[Tuple[str, str, int]]]]]:
    """Fallback default patterns if JSON file is not available"""
    return [
        {
            'name': 'Pattern 1: MPTCP Basic',
            'flows': [
                ('h1', 'h4', 5001),
                ('h3', 'h6', 5002),
                ('h5', 'h2', 5003),
                ('h11', 'h14', 5004),
                ('h13', 'h16', 5005),
                ('h15', 'h12', 5006)
            ]
        }
    ]

def run_iperf_pattern(net, pattern_name, flows, duration=1000, bandwidth='10M'):
    """Run iperf tests for a specific pattern with bandwidth limitation for h11-h16 only"""
    info(f'\n*** Starting Pattern: {pattern_name}\n')
    info(f'*** Duration: {duration} seconds\n')
    info(f'*** Bandwidth limit for h11-h16: {bandwidth}\n')
    
    # Define hosts that need bandwidth limitation (non-MPTCP hosts)
    limited_hosts = ['h11', 'h12', 'h13', 'h14', 'h15', 'h16']
    
    # Start iperf servers on receiving hosts
    servers = set()
    for sender, receiver, port in flows:
        if receiver not in servers:
            receiver_host = net.get(receiver)
            cmd = f'iperf -s -p {port} > /dev/null 2>&1 &'
            receiver_host.cmd(cmd)
            time.sleep(3)
            servers.add(receiver)
            info(f'Started iperf server on {receiver} port {port}\n')
            time.sleep(3)
    
    time.sleep(2)  # Give servers time to start
    
    # Start iperf clients with conditional bandwidth limitation
    for sender, receiver, port in flows:
        sender_host = net.get(sender)
        receiver_host = net.get(receiver)
        receiver_ip = receiver_host.IP()
        
        # Only add bandwidth limitation for h11-h16 (TCP only hosts)
        if sender in limited_hosts:
            cmd = f'iperf -c {receiver_ip} -p {port} -t {duration} -b {bandwidth} > /dev/null 2>&1 &'
            info(f'Started iperf client: {sender} -> {receiver} (port {port}, bandwidth: {bandwidth})\n')
        else:
            cmd = f'iperf -c {receiver_ip} -p {port} -t {duration} > /dev/null 2>&1 &'
            info(f'Started iperf client: {sender} -> {receiver} (port {port}, unlimited bandwidth)\n')
        
        sender_host.cmd(cmd)
        time.sleep(3)
    
    # Wait for duration
    info(f'*** Running traffic for {duration} seconds...\n')
    time.sleep(duration + 60)  # Extra time for cleanup
    
    # Kill all iperf processes
    for host in net.hosts:
        host.cmd('killall -9 iperf > /dev/null 2>&1')
    
    info(f'*** Pattern {pattern_name} completed\n')
    time.sleep(5)  # Brief pause between patterns

def run_iperf_pattern_custom_bandwidth(net, pattern_name, flows, duration=1000, bandwidth_per_flow=None):
    """
    Run iperf tests with custom bandwidth for each flow
    bandwidth_per_flow: dict mapping (sender, receiver) to bandwidth string, e.g., ('h1', 'h4'): '10M'
    Note: By default, only h11-h16 get bandwidth limitation
    """
    info(f'\n*** Starting Pattern: {pattern_name} (Custom Bandwidth)\n')
    info(f'*** Duration: {duration} seconds\n')
    
    # Define hosts that need bandwidth limitation by default (non-MPTCP hosts)
    limited_hosts = ['h11', 'h12', 'h13', 'h14', 'h15', 'h16']
    
    # Start iperf servers on receiving hosts
    servers = set()
    for sender, receiver, port in flows:
        if receiver not in servers:
            receiver_host = net.get(receiver)
            cmd = f'iperf -s -p {port} > /dev/null 2>&1 &'
            receiver_host.cmd(cmd)
            servers.add(receiver)
            info(f'Started iperf server on {receiver} port {port}\n')
    
    time.sleep(2)  # Give servers time to start
    
    # Start iperf clients with custom bandwidth
    for sender, receiver, port in flows:
        sender_host = net.get(sender)
        receiver_host = net.get(receiver)
        receiver_ip = receiver_host.IP()
        
        # Check if custom bandwidth is specified
        if bandwidth_per_flow and (sender, receiver) in bandwidth_per_flow:
            bw = bandwidth_per_flow[(sender, receiver)]
            cmd = f'iperf -c {receiver_ip} -p {port} -t {duration} -b {bw} > /dev/null 2>&1 &'
            info(f'Started iperf client: {sender} -> {receiver} (port {port}, bandwidth: {bw})\n')
        elif sender in limited_hosts:
            # Default bandwidth limitation for h11-h16
            cmd = f'iperf -c {receiver_ip} -p {port} -t {duration} -b 10M > /dev/null 2>&1 &'
            info(f'Started iperf client: {sender} -> {receiver} (port {port})\n')
        else:
            # No bandwidth limitation for MPTCP hosts
            cmd = f'iperf -c {receiver_ip} -p {port} -t {duration} > /dev/null 2>&1 &'
            info(f'Started iperf client: {sender} -> {receiver} (port {port}, unlimited bandwidth)\n')
        
        sender_host.cmd(cmd)
    
    # Wait for duration
    info(f'*** Running traffic for {duration} seconds...\n')
    time.sleep(duration + 60)  # Extra time for cleanup
    
    # Kill all iperf processes
    for host in net.hosts:
        host.cmd('killall -9 iperf > /dev/null 2>&1')
    
    info(f'*** Pattern {pattern_name} completed\n')
    time.sleep(5)  # Brief pause between patterns


def main(bandwidth='10M'):
    """Main test execution"""
    
    # Create topology
    topo = ThreeLayerTopo()
    remote_controller = RemoteController('c0', ip='172.29.25.79', port=6653)
    net = Mininet(topo=topo, switch=OVSKernelSwitch, link=TCLink, controller=remote_controller)
    
    info('*** Starting network\n')
    net.start()
    time.sleep(30)
    # Configure MPTCP and IPs as in original script
    # h1-h6: MPTCP enabled 
    # h11-h16: MPTCP disabled 
    for i in range(1, 7):
        h = net.get(f'h{i}')
        h.cmd('sysctl -w net.mptcp.enabled=1')
        h.setIP(f'10.0.1.{i}/24', intf=f'h{i}-eth1')
        h.setIP(f'10.0.2.{i}/24', intf=f'h{i}-eth2')
    
    for i in range(11, 17):
        h = net.get(f'h{i}')
        h.cmd('sysctl -w net.mptcp.enabled=0')
        h.setIP(f'10.0.3.{i-10}/24', intf=f'h{i}-eth1')
    
    time.sleep(5)  # Let network stabilize
    
    # Define 10 different traffic patterns
    # Pattern format: (sender, receiver, port)
    # Z1->Z2, Z2->Z3, Z3->Z1 with odd sending to even
    
    '''patterns = [
        # Pattern 1: Basic MPTCP hosts only
        {
            'name': 'Pattern 1: MPTCP Basic',
            'flows': [
                ('h1', 'h4', 5001),   # Z1->Z2
                ('h3', 'h6', 5002),   # Z2->Z3
                ('h5', 'h2', 5003),   # Z3->Z1
                ('h11', 'h14', 5004),
                ('h13', 'h16', 5005),
                ('h15', 'h12', 5006)
            ]
        },
        
        # Pattern 2: Non-MPTCP hosts only
        {
            'name': 'Pattern 2: Non-MPTCP Basic',
            'flows': [
                ('h11', 'h14', 5001),  # Z1->Z2
                ('h13', 'h16', 5002),  # Z2->Z3
                ('h15', 'h12', 5003),  # Z3->Z1
            ]
        },
        
        # Pattern 3: Mixed MPTCP and non-MPTCP
        {
            'name': 'Pattern 3: Mixed Traffic 1',
            'flows': [
                ('h1', 'h14', 5001),   # Z1->Z2 (MPTCP to non-MPTCP)
                ('h13', 'h6', 5002),   # Z2->Z3 (non-MPTCP to MPTCP)
                ('h5', 'h12', 5003),   # Z3->Z1 (MPTCP to non-MPTCP)
            ]
        },
        
        # Pattern 4: Reverse mixed
        {
            'name': 'Pattern 4: Mixed Traffic 2',
            'flows': [
                ('h11', 'h4', 5001),   # Z1->Z2 (non-MPTCP to MPTCP)
                ('h3', 'h16', 5002),   # Z2->Z3 (MPTCP to non-MPTCP)
                ('h15', 'h2', 5003),   # Z3->Z1 (non-MPTCP to MPTCP)
            ]
        },
        
        # Pattern 5: Multiple flows per zone
        {
            'name': 'Pattern 5: Multiple Flows 1',
            'flows': [
                ('h1', 'h4', 5001),    # Z1->Z2
                ('h11', 'h14', 5002),  # Z1->Z2
                ('h3', 'h6', 5003),    # Z2->Z3
                ('h13', 'h16', 5004),  # Z2->Z3
                ('h5', 'h2', 5005),    # Z3->Z1
                ('h15', 'h12', 5006),  # Z3->Z1
            ]
        },
        
        # Pattern 6: Cross pattern
        {
            'name': 'Pattern 6: Cross Traffic',
            'flows': [
                ('h1', 'h14', 5001),   # Z1->Z2 (cross)
                ('h11', 'h4', 5002),   # Z1->Z2 (cross)
                ('h3', 'h16', 5003),   # Z2->Z3 (cross)
                ('h13', 'h6', 5004),   # Z2->Z3 (cross)
                ('h5', 'h12', 5005),   # Z3->Z1 (cross)
                ('h15', 'h2', 5006),   # Z3->Z1 (cross)
            ]
        },
        
        # Pattern 7: Different port allocation
        {
            'name': 'Pattern 7: High Port Numbers',
            'flows': [
                ('h1', 'h4', 8001),    # Z1->Z2
                ('h3', 'h6', 8002),    # Z2->Z3
                ('h5', 'h2', 8003),    # Z3->Z1
                ('h11', 'h14', 8004),  # Z1->Z2
                ('h13', 'h16', 8005),  # Z2->Z3
                ('h15', 'h12', 8006),  # Z3->Z1
            ]
        },
        
        # Pattern 8: Asymmetric load
        {
            'name': 'Pattern 8: Asymmetric Load',
            'flows': [
                ('h1', 'h4', 5001),    # Z1->Z2
                ('h1', 'h14', 5002),   # Z1->Z2 (same sender)
                ('h3', 'h6', 5003),    # Z2->Z3
                ('h13', 'h6', 5004),   # Z2->Z3 (same receiver)
                ('h5', 'h2', 5005),    # Z3->Z1
                ('h15', 'h2', 5006),   # Z3->Z1 (same receiver)
            ]
        },
        
        # Pattern 9: Concentrated traffic
        {
            'name': 'Pattern 9: Concentrated Traffic',
            'flows': [
                ('h1', 'h4', 5001),    # All MPTCP hosts
                ('h3', 'h6', 5002),
                ('h5', 'h2', 5003),
                ('h1', 'h6', 5004),    # Additional cross-zone
                ('h3', 'h2', 5005),
                ('h5', 'h4', 5006),
            ]
        },
        
        # Pattern 10: Full mesh subset
        {
            'name': 'Pattern 10: Full Mesh Subset',
            'flows': [
                ('h1', 'h4', 5001),    # Z1->Z2
                ('h1', 'h14', 5002),   # Z1->Z2
                ('h11', 'h4', 5003),   # Z1->Z2
                ('h3', 'h6', 5004),    # Z2->Z3
                ('h3', 'h16', 5005),   # Z2->Z3
                ('h13', 'h6', 5006),   # Z2->Z3
                ('h5', 'h2', 5007),    # Z3->Z1
                ('h5', 'h12', 5008),   # Z3->Z1
                ('h15', 'h2', 5009),   # Z3->Z1
            ]
        },
    ]'''

    patterns = load_traffic_patterns()
    
    # Run all patterns
    info('\n*** Starting BigTopo Traffic Tests ***\n')
    info(f'*** Total patterns: {len(patterns)}\n')
    info(f'*** Duration per pattern: 1000 seconds\n')
    info(f'*** Bandwidth limit for h11-h16 (TCP only)\n')
    info(f'*** MPTCP hosts (h1-h6)\n')
    info(f'*** Total estimated time: {len(patterns) * 1000 / 60:.1f} minutes\n\n')

    start_time = time.time()
    
    for i, pattern in enumerate(patterns, 1):
        # [2025-06-26 19:31:09]
        timestamp = time.time()
        info(f'\n*** Starting test for pattern: {pattern["name"]} at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))}\n')
        info(f'{"="*60}\n')
        info(f'*** Running Test {i}/{len(patterns)}: {pattern["name"]}\n')
        info(f'{"="*60}\n')

        run_iperf_pattern(net, pattern['name'], pattern['flows'], duration=1000, bandwidth=bandwidth)
        time.sleep(15)
        
        elapsed = time.time() - start_time
        remaining = (len(patterns) - i) * 1000  # 1000 seconds per pattern
        info(f'\n*** Progress: {i}/{len(patterns)} patterns completed\n')
        info(f'*** Elapsed time: {elapsed/60:.1f} minutes\n')
        info(f'*** Estimated remaining: {remaining/60:.1f} minutes\n')
        timestamp = time.time()
        info(f'*** Finished test for pattern: {pattern["name"]} at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))}\n')
        info(f'{"="*60}\n')
    
    total_time = time.time() - start_time
    info(f'\n*** All tests completed!\n')
    info(f'*** Total time: {total_time/60:.1f} minutes\n')
    
    # Cleanup
    info('*** Stopping network\n')
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    
    # Check for bandwidth argument
    bandwidth = '10M'  # default
    if len(sys.argv) > 1:
        bandwidth = sys.argv[1]
        info(f'*** Using custom bandwidth for h11-h16: {bandwidth}\n')
    
    info('*** BigTopo Traffic Engineering Test Suite\n')
    info('*** This will run 10 different traffic patterns\n')
    info('*** Each pattern runs for 1000 seconds\n')
    info(f'*** Bandwidth limited to {bandwidth} for h11-h16 (TCP only)\n')
    info('*** MPTCP hosts (h1-h6) have unlimited bandwidth\n')
    info('*** Total runtime: approximately 83 minutes\n')
    info('*** Usage: python3 bigtopo_test.py [bandwidth]\n')
    info('*** Example: python3 bigtopo_test.py 10M\n\n')
    
    try:
        main(bandwidth)
    except KeyboardInterrupt:
        info('\n*** Test interrupted by user\n')
        os.system('killall -9 iperf > /dev/null 2>&1')
    except Exception as e:
        info(f'\n*** Error: {e}\n')
        os.system('killall -9 iperf > /dev/null 2>&1')