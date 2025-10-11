# setting.py

# Details handlers
SHOW_DELAY_STATS = False
SHOW_PATH_STATS = False
SHOW_FLOW_STATS = False
SHOW_UTILIZATION_STATS = False
FLOW_RATE_DEBUGGING = False

# MHG - Multi-Host Gateway configuration
MHG_LOGGING = False

# NM - Network Monitor configuration
NM_LOGGING = False

## FLM configuration
FLM_SHOW_REWARDS = False
FLM_SHOW_FLOW_STATS = False
FLM_SHOW_DETAILED_DEBUGGING = False
FLM_SHOW_DETAILED_DEBUGGING_MPTCP_CALCULATION = False
FLM_UPDATE_INTERVAL = 2  # seconds
FLM_COOKIE_EXPIRATION_TIME = 60  # seconds
FLM_MPTCP_FLOW_EXPIRATION_TIME = 2  # seconds



# LOG debugging
LINK_LOGGING = False
PATH_STATS_LOGGING = False
AGENT1_LOGGING = False
AGENT2_LOGGING = False
H_G_PPO_LOGGING = False


# USE AGENTS
USE_BASELINE = False # baseline agent RANDOM
USE_BASELINE2 = False
USE_BASELINE3 = False # all traffic goes through one path
USE_AGENT1 = True
USE_AGENT2 = True

# AGENT1 configuration
AGENT1_NEW_MODEL = True
AGENT1_SAVE_MODEL = True
AGENT1_MODEL_NAME = 'models/agent1/FlowSatisfier_model.pt'       # Specific model name to load (optional)

# AGENT2 configuration
AGENT2_LOAD_MODEL = False
AGENT2_SAVE_MODEL = True
AGENT2_NEW_MODEL = True
AGENT2_MODEL_NAME = 'models/agent2/GHOST_MAPPO_model.pt'       # Specific model name to load (optional)

# Agent2 Network Size
AGENT2_NETWORK_SIZE = "small"  # "small", "medium", "large"

# Agent2 Intervals
AGENT2_UPDATE_INTERVAL = 120  # 2 minutes
AGENT2_SAVE_INTERVAL = 300    # 5 minutes

# Agent2 Features (for future enhancement)
AGENT2_USE_ENHANCED_FEATURES = False
AGENT2_GNN_TYPE = "graphsage"  # "graphsage", "gat", "gcn"

# Agent2 Reward Settings
AGENT2_REWARD_TYPE = "threshold"  # "simple", "normalized", "threshold", "weighted"
AGENT2_MLU_THRESHOLDS = {
    'excellent': 0.5,
    'good': 0.7,
    'poor': 0.9
}

# Agent2 Training
AGENT2_BATCH_SIZE = 32
AGENT2_LEARNING_RATE = 3e-4
AGENT2_MEMORY_SIZE = 1000

# FLOW_RATE update
CONTINUE_UPDATE = True
MPTCP_KEEP_ALIVE_INTERVAL = 30  # seconds


# Number of paths to compute with Yen's algorithm
k = 2

# Scheduler interval in seconds (5 minutes)
scheduler_interval = 300

# Additional variables that could be made configurable:
# Default bandwidth and delay for links
default_bandwidth = 100
default_delay = 1
DEFAULT_FLOW_DEMAND = 15 # Mbps


# Logging configuration
log_level = "INFO"

# ARP EtherType
ETH_TYPE_ARP = 0x0806

# Priorities for different types of traffic
priority = {
    "arp": 1,
    "ipv4": 1,
    "MP_JOIN": 900,
    "default": 1000
}
PRIORITY_ARP = 1
PRIORITY_IPV4 = 1
PRIORITY_MP_JOIN = 900
PRIORITY_DEFAULT = 1000
PRIORITY_MPTCP = 950
PRIORITY_TCP = 850
PRIORITY_UDP = 700
DEFAULT_PRIORITY = 100

# Time to live for default flow entries
IDL_TIME = 5 # seconds
# hard timeout for default flow entries; 0 means no timeout
HARD_TIMEOUT = 0


#######################
## Network Monitor   ##
#######################

# Interval for monitoring links
LINK_MONITOR_INTERVAL = 1 # seconds
MPTCP_FLOW_DELETE_INTERVAL = 60 # seconds

# Link capacity
#LINK_MAX_CAPACITY = 10_000_000  # 1 Mbps
LINK_MAX_CAPACITY = 10000000  # 10 Mbps

#######################
## Network Delay     ##
#######################

DELAY_INTERVAL = 5  # seconds

#######################
## listener ID       ##
#######################

LISTENER_ID = 4999
LISTENER_PRIORITY = 1500

#######################
## flow Id           ##
#######################
COOKIE_ID = 0x0
MAX_COOKIE_ID = 0x7FFFFFFF
#######################
## flpw rate limit   ##
#######################

FLOW_RATE_INTERVAL = 1  # seconds
CHECKER_INTERVAL = 3  # seconds
FLOW_DEMAND = 15000000  # 15 Mbps
FLOW_DEMAND_INTERVAL = 30  # seconds
MPTCP_FLOW_DEL_INTERVAL = 90  # seconds
DEFAULT_FLOW_DEMAND = 10000000  # 10 Mbps


AGENT_UPDATE_INTERVAL = 20  # seconds


"""
Configuration settings for SDN Network Monitoring and Control
"""

# API Configuration
API_ENABLED = True
WEB_INTERFACE_ENABLED = True

# Server Configuration
API_HOST = '0.0.0.0'
API_PORT = 8080
DEBUG_MODE = False

# Module Configuration
ENABLE_TOPOLOGY_MANAGER = True
ENABLE_NETWORK_MONITOR = True
ENABLE_FLOW_RATE_MONITOR = True
ENABLE_DELAY_DETECTOR = True
ENABLE_FORWARDING = True

# Agent Configuration
ENABLE_AGENT1 = True
ENABLE_AGENT2 = True
ENABLE_BASELINE_MODE = False

# Monitoring Configuration
UPDATE_INTERVAL = 10  # seconds
STATS_COLLECTION_INTERVAL = 5  # seconds

# Web Interface Configuration
WEB_TEMPLATE_PATH = 'web/templates'
WEB_STATIC_PATH = 'web/static'

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Security Configuration
ENABLE_CORS = True
ALLOWED_ORIGINS = ['*']  # Configure specific origins in production

# Performance Configuration
MAX_FLOW_ENTRIES = 10000
MAX_PATH_HISTORY = 1000
CACHE_TIMEOUT = 300  # seconds


# Web Log Viewer Configuration
LOG_VIEWER_PORT = 8181           # Port for the web log viewer
MAX_LOG_ENTRIES = 2000          # Maximum number of log entries to keep in memory
WEB_LOG_VIEWER_ENABLED = True   # Enable/disable the web log viewer
#LOG_VIEWER_TOKEN = "1234567890abcdef"  # Token for accessing the log viewer
ALLOWED_LOGGERS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']  # Allowed log levels for viewing

TOPOLOGY = "BigTopo"  # Default topology to use