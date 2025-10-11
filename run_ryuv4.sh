#!/bin/bash

# Run Ryu with the required thigs
echo "Starting Ryu Controller..."
ryu-manager --observe-links ryu.app.ofctl_rest ryu.topology.switches test3/topology_manager.py test3/network_monitor.py test3/forwarding.py test3/arp_handler.py test3/RESTAPI.py
