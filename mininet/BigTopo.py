#!/usr/bin/env python3
"""
3-Layer Network Topology for Mininet
Implements the traffic engineering design with 3-hop, 4-hop, and 5-hop paths

Network Design (clean sequential switch numbering):
- 6 hosts: h1,h2 (5-hop), h7,h8 (4-hop), h13,h14 (3-hop)
- 18 switches: s1-s18 (sequential numbering)

Switch Mapping (logical sequential numbering):
Layer 2 (Access):     s1, s2, s3, s4, s5, s6
Layer 3 (Aggregation): s7, s8, s9, s10, s11, s12  
Layer 3 Final:        s13, s14, s15, s16, s17, s18

Path Examples:
- 5-hop: h1 → s1 → s7 → s13 → s15 → s17
- 4-hop: h7 → s3 → s9 → s15 → s17  
- 3-hop: h13 → s5 → s11 → s17
"""

from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.topo import Topo
from mininet.link import TCLink, Intf

class ThreeLayerTopo(Topo):
    """3-Layer Network Topology with Traffic Engineering"""
    
    def build(self):
        info('*** Building 3-Layer Network Topology\n')
        
        # Create hosts
        info('*** Adding hosts\n')
        # Zone 1 (high capacity traffic)
        h1 = self.addHost('h1', ip='10.0.1.1/24', mac='00:00:00:00:01:01')
        h11 = self.addHost('h11', ip='10.0.3.1/24', mac='00:00:00:00:11:01')
        h2 = self.addHost('h2', ip='10.0.1.2/24', mac='00:00:00:00:02:01')
        h12 = self.addHost('h12', ip='10.0.3.2/24', mac='00:00:00:00:12:01')

        # Zone 2 (high capacity traffic)
        h3 = self.addHost('h3', ip='10.0.1.3/24', mac='00:00:00:00:03:01')
        h13 = self.addHost('h13', ip='10.0.3.3/24', mac='00:00:00:00:13:01')
        h4 = self.addHost('h4', ip='10.0.1.4/24', mac='00:00:00:00:04:01')
        h14 = self.addHost('h14', ip='10.0.3.4/24', mac='00:00:00:00:14:01')

        # Zone 3 (high capacity traffic)
        h5 = self.addHost('h5', ip='10.0.1.5/24', mac='00:00:00:00:05:01')
        h15 = self.addHost('h15', ip='10.0.3.5/24', mac='00:00:00:00:15:01')
        h6 = self.addHost('h6', ip='10.0.1.6/24', mac='00:00:00:00:06:01')
        h16 = self.addHost('h16', ip='10.0.3.6/24', mac='00:00:00:00:16:01')

        # Layer 2: Access Switches (clean sequential numbering s1-s6)
        info('*** Adding Layer 2 (Access) switches\n')
        s1 = self.addSwitch('s1')    # Connected to h1 (5-hop path)
        s2 = self.addSwitch('s2')    # Connected to h2 (5-hop path)
        s3 = self.addSwitch('s3')    # Connected to h7 (4-hop path)
        s4 = self.addSwitch('s4')    # Connected to h8 (4-hop path)
        s5 = self.addSwitch('s5')    # Connected to h13 (3-hop path)
        s6 = self.addSwitch('s6')    # Connected to h14 (3-hop path)
        
        # Layer 3: Aggregation Switches (sequential numbering s7-s12)
        info('*** Adding Layer 3 (Aggregation) switches\n')
        s7 = self.addSwitch('s7')    # For 5-hop paths (from s1)
        s8 = self.addSwitch('s8')    # For 5-hop paths (from s2)
        s9 = self.addSwitch('s9')    # For 4-hop paths (from s3)
        s10 = self.addSwitch('s10')  # For 4-hop paths (from s4)
        s11 = self.addSwitch('s11')  # For 3-hop paths (from s5)
        s12 = self.addSwitch('s12')  # For 3-hop paths (from s6)
        
        # Layer 3 Final: Core/Endpoint Switches (sequential numbering s13-s18)
        info('*** Adding Layer 3 Final (Core/Endpoint) switches\n')
        s13 = self.addSwitch('s13')  # Intermediate for 5-hop paths
        s14 = self.addSwitch('s14')  # Intermediate for 5-hop paths
        s15 = self.addSwitch('s15')  # Aggregation core
        s16 = self.addSwitch('s16')  # Aggregation core
        s17 = self.addSwitch('s17')  # Final endpoint
        s18 = self.addSwitch('s18')  # Final endpoint
        
        # Host to Access Switch connections
        info('*** Creating host-to-switch links\n')
        # 5-hop path connections
        self.addLink(h1, s1, port1=1, port2=4, cls=TCLink, bw=10, addr1='00:00:00:00:01:02')
        self.addLink(h1, s1, port1=2, port2=5, cls=TCLink, bw=10, addr1='00:00:00:00:01:03')
        self.addLink(h11, s1, port1=1, port2=6, cls=TCLink, bw=10, addr1='00:00:00:00:11:02')
        self.addLink(h2, s2, port1=1, port2=4, cls=TCLink, bw=10, addr1='00:00:00:00:02:02')
        self.addLink(h2, s2, port1=2, port2=5, cls=TCLink, bw=10, addr1='00:00:00:00:02:03')
        self.addLink(h12, s2, port1=1, port2=6, cls=TCLink, bw=10, addr1='00:00:00:00:12:02')

        # 4-hop path connections
        self.addLink(h3, s3, port1=1, port2=4, cls=TCLink, bw=10, addr1='00:00:00:00:03:02')
        self.addLink(h3, s3, port1=2, port2=5, cls=TCLink, bw=10, addr1='00:00:00:00:03:03')
        self.addLink(h13, s3, port1=1, port2=6, cls=TCLink, bw=10, addr1='00:00:00:00:13:02')
        self.addLink(h4, s4, port1=1, port2=4, cls=TCLink, bw=10, addr1='00:00:00:00:04:02')
        self.addLink(h4, s4, port1=2, port2=5, cls=TCLink, bw=10, addr1='00:00:00:00:04:03')
        self.addLink(h14, s4, port1=1, port2=6, cls=TCLink, bw=10, addr1='00:00:00:00:14:02')

        # 3-hop path connections
        self.addLink(h5, s5, port1=1, port2=4, cls=TCLink, bw=10, addr1='00:00:00:00:05:02')
        self.addLink(h5, s5, port1=2, port2=5, cls=TCLink, bw=10, addr1='00:00:00:00:05:03')
        self.addLink(h15, s5, port1=1, port2=6, cls=TCLink, bw=10, addr1='00:00:00:00:15:02')
        self.addLink(h6, s6, port1=1, port2=4, cls=TCLink, bw=10, addr1='00:00:00:00:06:02')
        self.addLink(h6, s6, port1=2, port2=5, cls=TCLink, bw=10, addr1='00:00:00:00:06:03')
        self.addLink(h16, s6, port1=1, port2=6, cls=TCLink, bw=10, addr1='00:00:00:00:16:03')


        # Access to Aggregation connections
        info('*** Creating access-to-aggregation links\n')
        # For 5-hop paths (h1, h2)
        self.addLink(s1, s7, port1=1, port2=1, cls=TCLink, bw=10, delay='5ms')
        self.addLink(s1, s8, port1=2, port2=1, cls=TCLink, bw=10, delay='5ms')
        self.addLink(s2, s7, port1=1, port2=2, cls=TCLink, bw=10, delay='5ms')
        self.addLink(s2, s8, port1=2, port2=2, cls=TCLink, bw=10, delay='5ms')

        # For 4-hop paths (h7, h8)
        self.addLink(s3, s9, port1=1, port2=1, cls=TCLink, bw=10, delay='5ms')
        self.addLink(s3, s10, port1=2, port2=1, cls=TCLink, bw=10, delay='5ms')
        self.addLink(s4, s9, port1=1, port2=2, cls=TCLink, bw=10, delay='5ms')
        self.addLink(s4, s10, port1=2, port2=2, cls=TCLink, bw=10, delay='5ms')

        # For 3-hop paths (h13, h14)
        self.addLink(s5, s11, port1=1, port2=1, cls=TCLink, bw=10, delay='5ms')
        self.addLink(s5, s12, port1=2, port2=1, cls=TCLink, bw=10, delay='5ms')
        self.addLink(s6, s11, port1=1, port2=2, cls=TCLink, bw=10, delay='5ms')
        self.addLink(s6, s12, port1=2, port2=2, cls=TCLink, bw=10, delay='5ms')

        # Aggregation to Core connections
        info('*** Creating aggregation-to-core links\n')
        # 5-hop path aggregation to intermediate core
        self.addLink(s7, s13, port1=3, port2=1, cls=TCLink, bw=10, delay='5ms')
        self.addLink(s8, s14, port1=3, port2=1, cls=TCLink, bw=10, delay='5ms')

        # 4-hop path aggregation to aggregation core
        self.addLink(s9, s15, port1=3, port2=1, cls=TCLink, bw=10, delay='5ms')
        self.addLink(s10, s16, port1=3, port2=1, cls=TCLink, bw=10, delay='5ms')

        # 3-hop path aggregation to final endpoints
        self.addLink(s11, s17, port1=3, port2=1, cls=TCLink, bw=10, delay='5ms')
        self.addLink(s12, s18, port1=3, port2=1, cls=TCLink, bw=10, delay='5ms')

        # Core interconnections
        info('*** Creating core interconnections\n')
        # Intermediate core to aggregation core
        self.addLink(s13, s15, port1=2, port2=2, cls=TCLink, bw=10, delay='5ms')
        self.addLink(s14, s16, port1=2, port2=2, cls=TCLink, bw=10, delay='5ms')

        # Aggregation core to final endpoints
        self.addLink(s15, s17, port1=3, port2=2, cls=TCLink, bw=10, delay='5ms')
        self.addLink(s16, s18, port1=3, port2=2, cls=TCLink, bw=10, delay='5ms')

        info('*** Topology creation complete\n')

def run_topology():
    """Run the 3-Layer topology with testing"""
    
    topo = ThreeLayerTopo()
    remote_controller = RemoteController('c0', ip='172.29.25.79', port=6653)
    net = Mininet(topo=topo, switch=OVSKernelSwitch, link=TCLink, controller=remote_controller)

    info('*** Starting network\n')
    net.start()
    
    h1 = net.get('h1')
    h2 = net.get('h2')
    h3 = net.get('h3')
    h4 = net.get('h4')
    h5 = net.get('h5')
    h6 = net.get('h6')

    h11 = net.get('h11')
    h12 = net.get('h12')
    h13 = net.get('h13')
    h14 = net.get('h14')
    h15 = net.get('h15')
    h16 = net.get('h16')


    h1.cmd('sysctl -w net.mptcp.enabled=1')
    h2.cmd('sysctl -w net.mptcp.enabled=1')
    h3.cmd('sysctl -w net.mptcp.enabled=1')
    h4.cmd('sysctl -w net.mptcp.enabled=1')
    h5.cmd('sysctl -w net.mptcp.enabled=1')
    h6.cmd('sysctl -w net.mptcp.enabled=1')
    h11.cmd('sysctl -w net.mptcp.enabled=0')
    h12.cmd('sysctl -w net.mptcp.enabled=0')
    h13.cmd('sysctl -w net.mptcp.enabled=0')
    h14.cmd('sysctl -w net.mptcp.enabled=0')
    h15.cmd('sysctl -w net.mptcp.enabled=0')
    h16.cmd('sysctl -w net.mptcp.enabled=0')

    h1.setIP('10.0.1.1/24', intf='h1-eth1')
    h1.setIP('10.0.2.1/24', intf='h1-eth2')
    h2.setIP('10.0.1.2/24', intf='h2-eth1')
    h2.setIP('10.0.2.2/24', intf='h2-eth2')
    h3.setIP('10.0.1.3/24', intf='h3-eth1')
    h3.setIP('10.0.2.3/24', intf='h3-eth2')
    h4.setIP('10.0.1.4/24', intf='h4-eth1')
    h4.setIP('10.0.2.4/24', intf='h4-eth2')
    h5.setIP('10.0.1.5/24', intf='h5-eth1')
    h5.setIP('10.0.2.5/24', intf='h5-eth2')
    h6.setIP('10.0.1.6/24', intf='h6-eth1')
    h6.setIP('10.0.2.6/24', intf='h6-eth2')


    h11.setIP('10.0.3.1/24', intf='h11-eth1')
    h12.setIP('10.0.3.2/24', intf='h12-eth1')
    h13.setIP('10.0.3.3/24', intf='h13-eth1')
    h14.setIP('10.0.3.4/24', intf='h14-eth1')
    h15.setIP('10.0.3.5/24', intf='h15-eth1')
    h16.setIP('10.0.3.6/24', intf='h16-eth1')



    # Start CLI for interactive testing
    info('*** Running CLI (type "exit" to quit)\n')
    info('*** Try these commands:\n')
    info('    pingall - test all connectivity\n')
    info('    h1 ping h13 - test cross-tier connectivity\n')
    info('    h1 traceroute h13 - see routing path\n')
    info('    iperf h1 h13 - test bandwidth\n')
    info('    sh ovs-ofctl dump-flows s1 - see switch flow tables\n')
    CLI(net)
    
    info('*** Stopping network\n')
    net.stop()




if __name__ == '__main__':
    setLogLevel('info')
    
    info('*** 3-Layer Traffic Engineering Network Topology\n')
    info('*** Clean sequential switch numbering (s1-s18)\n')
    info('*** This topology demonstrates QoS through path engineering\n')
    info('*** Different hosts get different path lengths for traffic prioritization\n\n')
    
    try:
        run_topology()
    except KeyboardInterrupt:
        info('\n*** Interrupted by user\n')
    except Exception as e:
        info(f'\n*** Error: {e}\n')



