#!/bin/bash

# update date and time
sudo ntpdate 0.us.pool.ntp.org

# Clear any existing Mininet state
sudo mn -c

# Run test.py with preserved environment variables 3M
sudo python3 test.py

# Clear any existing Mininet state
sudo mn -c
