# RL2/utils/reward_functions.py
"""
Reward calculation utilities
"""

import numpy as np
from typing import Optional, List
import torch

def calculate_reward1(prev_mlu, current_mlu, action, config,
                     invalid_levels: Optional[List[str]] = None):
    """
    Calculate reward based on MLU change and action
    
    Args:
        prev_mlu: Previous maximum link utilization
        current_mlu: Current maximum link utilization
        action: Action taken
        config: Agent configuration
        
    Returns:
        float: Calculated reward
    """
    reward_type = config.get('reward_type', 'threshold')
    penalty = config.get('invalid_penalty', 0.2) * len(invalid_levels or [])
    if reward_type == 'simple':
        return simple_reward(prev_mlu, current_mlu)
    elif reward_type == 'normalized':
        return normalized_reward(prev_mlu, current_mlu)
    elif reward_type == 'threshold':
        return threshold_reward(prev_mlu, current_mlu, config)
    elif reward_type == 'penalized':
        return -penalty
    else:
        return weighted_reward(prev_mlu, current_mlu, action, config)


def simple_reward(prev_mlu, current_mlu):
    """Simple difference reward"""
    return prev_mlu - current_mlu


def normalized_reward(prev_mlu, current_mlu):
    """Normalized improvement reward"""
    if prev_mlu > 0:
        return (prev_mlu - current_mlu) / prev_mlu
    return 0.0

def calculate_reward(prev_mlu, current_mlu, action, config,
                     invalid_levels: Optional[List[str]] = None):
    """
    Calculate reward based on MLU change and action
    
    Args:
        prev_mlu: Previous maximum link utilization
        current_mlu: Current maximum link utilization
        action: Action taken
        config: Agent configuration
        invalid_levels: List of levels that were invalid during action selection
        
    Returns:
        float: Calculated reward
    """
    reward_type = config.get('reward_type', 'threshold')
    
    # Calculate base reward
    if reward_type == 'simple':
        base_reward = simple_reward(prev_mlu, current_mlu)
    elif reward_type == 'normalized':
        base_reward = normalized_reward(prev_mlu, current_mlu)
    elif reward_type == 'threshold':
        base_reward = threshold_reward(prev_mlu, current_mlu, config)
    elif reward_type == 'penalized':
        base_reward = -config.get('invalid_penalty', 0.2)
    else:
        base_reward = weighted_reward(prev_mlu, current_mlu, action, config)
    
    # Apply penalties for invalid levels
    if invalid_levels:
        penalty_per_level = config.get('invalid_penalty', 0.2)
        total_penalty = penalty_per_level * len(invalid_levels)
        base_reward -= total_penalty
        
        # Log which levels were invalid
        if config.get('logging', False):
            print(f"Applied penalty for invalid levels {invalid_levels}: -{total_penalty}")
    
    return base_reward

def threshold_reward(prev_mlu, current_mlu, config):
    """Threshold-based reward with improvement bonus"""
    thresholds = config.get('mlu_thresholds', {
        'excellent': 0.5,
        'good': 0.7,
        'poor': 0.9
    })
    
    # Base reward based on current MLU
    if current_mlu < thresholds['excellent']:
        base_reward = 1.0
    elif current_mlu < thresholds['good']:
        base_reward = 0.5
    elif current_mlu < thresholds['poor']:
        base_reward = -0.5
    else:
        base_reward = -1.0
    
    # Improvement bonus
    improvement = prev_mlu - current_mlu
    if improvement > 0:
        bonus = min(improvement * 2, 0.5)  # Cap bonus at 0.5
    else:
        bonus = max(improvement * 2, -0.5)  # Cap penalty at -0.5
    
    return base_reward + bonus


def weighted_reward(prev_mlu, current_mlu, action, config):
    """
    Weighted multi-objective reward
    
    Components:
    - MLU improvement
    - Network stability
    - Migration cost
    """
    weights = config.get('reward_weights', {
        'mlu': 0.6,
        'improvement': 0.3,
        'stability': 0.1
    })
    
    # MLU component
    mlu_reward = threshold_reward(prev_mlu, current_mlu, config)
    
    # Improvement component
    improvement = normalized_reward(prev_mlu, current_mlu)
    
    # Stability component (penalize if action was exploratory)
    if action.get('exploration', False):
        stability = -0.1
    else:
        stability = 0.1
    
    # Weighted sum
    total_reward = (
        weights['mlu'] * mlu_reward +
        weights['improvement'] * improvement +
        weights['stability'] * stability
    )
    
    return np.clip(total_reward, -2.0, 2.0)


def flow_fairness_reward(flow_rates):
    """
    Calculate Jain's fairness index for flow rates
    
    Args:
        flow_rates: List of flow rates
        
    Returns:
        float: Fairness index (0 to 1)
    """
    if not flow_rates or len(flow_rates) == 0:
        return 0.0
    
    n = len(flow_rates)
    sum_rates = sum(flow_rates)
    sum_squares = sum(r**2 for r in flow_rates)
    
    if sum_squares == 0:
        return 0.0
    
    fairness = (sum_rates ** 2) / (n * sum_squares)
    return fairness


def migration_cost_penalty(flow_size, num_hops):
    """
    Calculate migration cost penalty
    
    Args:
        flow_size: Size of flow being migrated
        num_hops: Number of hops in new path
        
    Returns:
        float: Negative penalty value
    """
    # Normalize flow size (assuming max 1 Gbps)
    normalized_size = min(flow_size / 1000.0, 1.0)
    
    # Normalize hop count (assuming max 10 hops)
    normalized_hops = min(num_hops / 10.0, 1.0)
    
    # Penalty increases with flow size and hop count
    penalty = -(normalized_size * 0.7 + normalized_hops * 0.3) * 0.2
    
    return penalty

def r_switch1(prev_mlu, new_mlu):          return prev_mlu - new_mlu
def r_port1(util, is_non_bottleneck):      return 1.0 if is_non_bottleneck else -1.0
def r_flow1(bps, cap):                     return bps / cap                 # 0-1
def r_newport1(util_before, util_after):   return util_before - util_after


def switch_reward(mlu_prev, mlu_after, old_m, new_m):
    '''Calculate switch reward based on MLU change and link utilization.
       ( mlu_prev - mlu_after ) / mlu_prev
       normalized to [-1, 1]
    Args:
        mlu_prev (float): Previous maximum link utilization.
        mlu_after (float): Current maximum link utilization.
        link_utilization_before (float): Link utilization before the action.
        link_utilization_after (float): Link utilization after the action.
    '''
    alpha = 0.75 # Weight for MLU change
    reward = - ( new_m - old_m ) * alpha
    return min(max(reward, -1.0), 1.0)

# reward_functions.py
  
def switch_reward_old(mlu_prev, mlu_after):
    '''Calculate switch reward based on MLU change.
       ( mlu_prev - mlu_after ) / mlu_prev
       normalized to [-1, 1]
    Args:
        mlu_prev (float): Previous maximum link utilization.
        mlu_after (float): Current maximum link utilization.
    Returns:
        float: Reward value.
    '''
    if mlu_prev == 0: return 0.0
    return min(max((mlu_prev - mlu_after) / mlu_prev, -1.0), 1.0)

def port_reward(sel_util, max_util_on_sw):
    '''Calculate port reward based on selected utilization.
       If selected utilization is >= 90% of max utilization, return 1.0,
       otherwise return -1.0.
    Args:
        sel_util (float): Selected port utilization.
        max_util_on_sw (float): Maximum utilization on the switch.
    Returns:
        float: Reward value.
    '''
    if max_util_on_sw == 0: return 0.0
    r = 2 * (sel_util / max_util_on_sw) - 1
    return float(min(max(r, -1.0), 1.0))
    # return 1.0 if sel_util >= 0.65 * max_util_on_sw else -1.0

def flow_reward(rate_bps, link_capacity_bps, dst_reachable=True):
    '''Calculate flow reward based on rate and link capacity.
       If destination is reachable, return the ratio of rate to capacity,
       capped at 1.0. If not reachable, return -1.0.
    Args:
        rate_bps (float): Flow rate in bits per second.
        link_capacity_bps (float): Link capacity in bits per second.
        dst_reachable (bool): Whether the destination is reachable.
    Returns:
        float: Reward value, normalized to [0, 1] or -1.0 if not reachable.
    '''
    if not dst_reachable:
        return -0.1
    
    reward = (rate_bps / link_capacity_bps) * 2 - 1
    return float(np.clip(reward, -1.0, 1.0))

def newport_reward(src_util_before, util_before, util_after = 0.0):
    '''Calculate new port reward based on utilization change.
       ( src_util_before - util_before ) / src_util_before
       normalized to [-1, 1]
    Args:
        util_before (float): Utilization before the action.
        util_after (float): Utilization after the action.
    Returns:
        float: Reward value.
    '''
    if src_util_before == 0: return 0.0
    r = ((src_util_before - util_before) / src_util_before)
    return float(min(max(r, -1.0), 1.0))


def enhanced_switch_reward(mlu_prev, mlu_after, switch_mlus_before, switch_mlus_after):
    """
    Enhanced switch reward considering both MLU improvement and load balance
    
    Args:
        mlu_prev: Previous network MLU
        mlu_after: Current network MLU
        switch_mlus_before: Dict of switch MLUs before action
        switch_mlus_after: Dict of switch MLUs after action
    
    Returns:
        float: Reward value [-1, 1]
    """
    # MLU improvement component (70% weight)
    if mlu_prev == 0:
        mlu_component = 0.0
    else:
        mlu_component = max((mlu_prev - mlu_after) / mlu_prev, -1.0)
    
    # Load balance component (30% weight) - lower std dev is better
    std_before = np.std(list(switch_mlus_before.values())) if len(switch_mlus_before) > 1 else 0
    std_after = np.std(list(switch_mlus_after.values())) if len(switch_mlus_after) > 1 else 0
    
    if std_before == 0:
        balance_component = 0.0
    else:
        balance_component = min((std_before - std_after) / std_before, 1.0)
    
    return 0.7 * mlu_component + 0.3 * balance_component


def enhanced_port_reward(sel_util, max_util_on_sw, port_utils_on_switch):
    """
    Enhanced port reward considering utilization and load distribution
    
    Args:
        sel_util: Selected port utilization
        max_util_on_sw: Maximum utilization on the switch
        port_utils_on_switch: List of all port utilizations on the switch
    
    Returns:
        float: Reward value [-1, 1]
    """
    # High utilization selection (60% weight)
    util_threshold = 0.8  # Lower threshold from 0.9
    util_component = 1.0 if sel_util >= util_threshold * max_util_on_sw else -0.5
    
    # Port is among top utilized (40% weight)
    sorted_utils = sorted(port_utils_on_switch, reverse=True)
    if len(sorted_utils) > 2 and sel_util in sorted_utils[:3]:
        rank_component = 0.5
    else:
        rank_component = -0.5
    
    return 0.6 * util_component + 0.4 * rank_component


def enhanced_flow_reward(rate_bps, link_capacity_bps, dst_reachable, 
                        flow_rates_before, flow_rates_after):
    """
    Enhanced flow reward considering rate, reachability, and fairness
    
    Args:
        rate_bps: Selected flow rate
        link_capacity_bps: Link capacity
        dst_reachable: Whether destination is reachable
        flow_rates_before: All flow rates before action
        flow_rates_after: All flow rates after action
    
    Returns:
        float: Reward value [-1, 1]
    """
    if not dst_reachable:
        return -1.0
    
    # Flow size component (40% weight) - prefer larger flows for migration
    rate_component = min(rate_bps / link_capacity_bps, 1.0)
    
    # Flow selection strategy (30% weight) - prefer elephant flows
    if rate_bps > 0.5 * link_capacity_bps:  # Elephant flow
        size_bonus = 0.5
    elif rate_bps > 0.1 * link_capacity_bps:  # Medium flow
        size_bonus = 0.2
    else:  # Mice flow
        size_bonus = -0.3
    
    # Fairness improvement (30% weight)
    fairness_before = calculate_jains_fairness(flow_rates_before)
    fairness_after = calculate_jains_fairness(flow_rates_after)
    fairness_component = (fairness_after - fairness_before) * 2  # Scale to [-1, 1]
    
    return 0.4 * rate_component + 0.3 * size_bonus + 0.3 * fairness_component


def calculate_jains_fairness(flow_rates):
    """Calculate Jain's fairness index"""
    if not flow_rates or len(flow_rates) == 0:
        return 0.0
    
    rates = [r for r in flow_rates if r > 0]  # Filter out zero rates
    if not rates:
        return 0.0
        
    n = len(rates)
    sum_rates = sum(rates)
    sum_squares = sum(r**2 for r in rates)
    
    if sum_squares == 0:
        return 0.0
    
    return (sum_rates ** 2) / (n * sum_squares)


# Update no_op_reward function in reward_functions.py:

def no_op_reward123(level, current_mlu, network_state, config):
    """
    Calculate reward for No-Op action at different hierarchy levels
    
    Args:
        level: Which level chose No-Op ('switch', 'port', 'flow', 'new_port')
        current_mlu: Current network MLU
        network_state: Current network state dict
        config: Agent configuration
        
    Returns:
        float: Reward value
    """
    # For switch/port levels - use MLU-based reward
    if level in ['switch', 'port']:
        # Get MLU thresholds
        thresholds = config.get('mlu_thresholds', {
            'excellent': 0.5,
            'good': 0.7,
            'poor': 0.9
        })
        
        # Reward for maintaining good state
        if current_mlu < thresholds['excellent']:
            return config.get('no_op_rewards', {}).get('excellent_state', 0.8)
        elif current_mlu < thresholds['good']:
            return config.get('no_op_rewards', {}).get('good_state', 0.3)
        elif current_mlu < thresholds['poor']:
            return config.get('no_op_rewards', {}).get('poor_state', -0.2)
        else:
            return config.get('no_op_rewards', {}).get('critical_state', -0.5)
    
    # For flow level - consider port utilization and flow characteristics
    elif level == 'flow':
        # Check if we selected a highly utilized port
        port_util = network_state.get('selected_port_util', 0.0)
        
        # If port is highly utilized, No-Op might be good to avoid congestion
        if port_util > 0.8:
            return config.get('no_op_rewards', {}).get('flow_congested_port', 0.2)
        elif port_util > 0.6:
            return config.get('no_op_rewards', {}).get('flow_moderate_port', -0.05)
        else:
            # Low utilized port - should probably select a flow
            return config.get('no_op_rewards', {}).get('flow_underutilized_port', -0.3)
    
    # For new_port level
    else:
        # Small negative to encourage completing the action if we got this far
        return config.get('no_op_rewards', {}).get('incomplete_action', -0.1)

def _no_op_reward(level: str,
                  pre_metrics: dict,
                  config: dict) -> float:
    """
    Compute reward for taking a No-op at a given hierarchy level.

    Parameters
    ----------
    level        : 'switch' | 'port' | 'flow' | 'new_port'
    pre_metrics  : dict of utilisation numbers captured *before* action
                   (keys: 'mlu', 'port_utils', 'flow_port_util', etc.)
    config       : full experiment config

    Returns
    -------
    r : float     (+ small, 0, or – penalty)
    """
    thr  = config['no_op_thresholds'][level]
    good = config['no_op_rewards']['good_state']
    bad  = config['no_op_rewards']['bad_state']

    if level == 'switch':
        metric = pre_metrics['mlu']
    elif level == 'port':
        metric = max(pre_metrics['port_utils'].values())
    elif level == 'flow':
        metric = pre_metrics['flow_port_util']
    else:  # new_port
        metric = pre_metrics['pred_new_port_util']

    return good if metric <= thr else bad


# Could be. 

def no_op_reward(level, current_mlu, network_state, config, action=None):
    """
    Enhanced No-Op reward calculation with context-aware design
    """
    # Get recent MLU history for trend analysis
    mlu_history = network_state.get('mlu_history', [current_mlu])
    mlu_trend = calculate_mlu_trend(mlu_history) if len(mlu_history) > 1 else 0
    print(f"[Reward_function][no_op_reward] Current MLU: {current_mlu}, MLU trend: {mlu_trend}")
    
    if level == 'switch':
        return switch_no_op_reward(current_mlu, mlu_trend, network_state, config)
    elif level == 'port':
        return port_no_op_reward(current_mlu, mlu_trend, network_state, config, action)
    elif level == 'flow':
        return flow_no_op_reward(current_mlu, network_state, config, action)
    elif level == 'new_port':
        return new_port_no_op_reward(network_state, config, action)


def switch_no_op_reward(current_mlu, mlu_trend, network_state, config):
    """
    Switch-level No-Op reward based on network state and trends
    """
    print(f"[Reward_function][switch_no_op_reward] Current MLU: {current_mlu}, MLU trend: {mlu_trend}")
    thresholds = config.get('mlu_thresholds', {
        'excellent': 0.5,
        'good': 0.7,
        'poor': 0.9
    })
    
    # Base reward from MLU state
    if current_mlu < thresholds['excellent']:
        base_reward = 0.8
    elif current_mlu < thresholds['good']:
        base_reward = 0.4
    elif current_mlu < thresholds['poor']:
        base_reward = -0.3
    else:
        base_reward = -0.8
    print(f"[Reward_function][switch_no_op_reward] Base reward: {base_reward}")
    # Trend modifier: reward stability, penalize worsening
    trend_modifier = 0.0
    if mlu_trend < -0.05:  # Improving
        trend_modifier = 0.1
    elif mlu_trend > 0.05:  # Worsening
        trend_modifier = -0.3
    print(f"[Reward_function][switch_no_op_reward] Trend modifier: {trend_modifier}")
    # Network balance bonus
    switch_mlus = network_state.get('switch_mlus', {})
    print(f"[Reward_function][switch_no_op_reward] Switch MLUs: {switch_mlus}")
    balance_bonus = 0.0
    if switch_mlus:
        std_dev = np.std(list(switch_mlus.values()))
        #balance_bonus = 0.2 if std_dev < 0.1 else 0.0
        if std_dev < 0.4 and std_dev > 0.2:
            balance_bonus = 0.1
    else:
        balance_bonus = 0.0
    print(f"[Reward_function][switch_no_op_reward] Balance bonus: {balance_bonus}")
    # Recent action penalty (avoid too frequent No-Ops)
    # It is zero for now. 
    recent_no_ops = network_state.get('recent_no_op_count', 0) 
    print(f"[Reward_function][switch_no_op_reward] Recent No-Op count: {recent_no_ops}")
    frequency_penalty = -0.1 * min(recent_no_ops, 3)
    print(f"[Reward_function][switch_no_op_reward] Frequency penalty: {frequency_penalty}")
    reward = base_reward + trend_modifier + balance_bonus + frequency_penalty
    print(f"[Reward_function][switch_no_op_reward] Final reward: {reward}")
    
    return np.clip(reward, -1.0, 1.0)


def port_no_op_reward(current_mlu, mlu_trend, network_state, config, action):
    """
    Port-level No-Op reward considering switch state
    """
    # Get selected switch MLU
    switch_mlu = action['validation'].get('switch', {}).get('selected_mlu', current_mlu)
    print(f"[Reward_function][port_no_op_reward] Selected switch MLU: {switch_mlu}, Current MLU: {current_mlu}")
    # If switch MLU is low, encourage No-Op
    if switch_mlu < 0.5:
        return 0.5
    elif switch_mlu < 0.:
        return 0.1
    else:
        # High MLU switch - penalize No-Op
        return -0.4


def flow_no_op_reward(current_mlu, network_state, config, action):
    """
    Flow-level No-Op reward with sophisticated logic
    """
    port_util = action['validation'].get('port', {}).get('utilization', 0.0)
    port_flow_count = action['validation'].get('port', {}).get('selected_flow_count', 0)
    print(f"[Reward_function][flow_no_op_reward] Port utilization: {port_util}, Flow count: {port_flow_count}")
    # Get candidate new ports info
    #candidate_utils = action.get('flow', {}).get('candidate_new_port_utils', [])
    # Get candidate new ports info - fix for No-Op case
    flow_data = action['validation'].get('flow', {})
    if isinstance(flow_data, dict):
        candidate_utils = flow_data.get('candidate_new_ports_utils', [])  # Note: typo in original key
    else:
        candidate_utils = []
    print(f"[Reward_function][flow_no_op_reward] Candidate port utils: {candidate_utils}")
    # Decision factors
    factors = {
        'port_congestion': 0.0,
        'alternative_quality': 0.0,
        'flow_count_penalty': 0.0,
        'mlu_state': 0.0
    }
    
    # Port congestion factor
    if port_util > 0.85:
        factors['port_congestion'] = 0.4  # Highly congested
    elif port_util > 0.7:
        factors['port_congestion'] = 0.1
    else:
        factors['port_congestion'] = -0.3  # Underutilized
    
    # Alternative quality factor
    if candidate_utils:
        best_alternative = min(candidate_utils)
        if best_alternative > 0.8:  # All alternatives bad
            factors['alternative_quality'] = 0.3
        elif best_alternative < 0.3:  # Good alternatives exist
            factors['alternative_quality'] = -0.4
    
    # Flow count factor
    if port_flow_count > 10:
        factors['flow_count_penalty'] = -0.2  # Many flows to choose from
    elif port_flow_count < 3:
        factors['flow_count_penalty'] = 0.1  # Few flows
    
    # MLU state factor
    if current_mlu < 0.5:
        factors['mlu_state'] = 0.2  # Network is healthy
    else:
        factors['mlu_state'] = -0.2  # Network needs help
    
    # Weighted sum
    weights = config.get('flow_no_op_weights', {
        'port_congestion': 0.4,
        'alternative_quality': 0.3,
        'flow_count_penalty': 0.1,
        'mlu_state': 0.2
    })
    
    total_reward = sum(factors[k] * weights.get(k, 0.25) for k in factors)
    return np.clip(total_reward, -1.0, 1.0)


def new_port_no_op_reward(network_state, config, action):
    """
    New port No-Op reward (usually discouraged)
    """
    # If we got this far, we should complete the action
    # Unless all new ports are terrible
    selected_flow_rate = action['validation'].get('flow', {}).get('selected_flow_rate', 0)
    print(f"[Reward_function][new_port_no_op_reward] Selected flow rate: {selected_flow_rate}")
    
    # Only consider No-Op if flow is very small
    if selected_flow_rate < 0.01 * config.get('link_capacity', 1000):
        return -0.05  # Small penalty for tiny flows
    else:
        return -0.5  # Larger penalty - should complete action


def calculate_mlu_trend(mlu_history, window=5):
    """Calculate MLU trend over recent history"""
    if len(mlu_history) < 2:
        return 0.0
    
    recent = mlu_history[-window:]
    if len(recent) < 2:
        return 0.0
    
    # Simple linear trend
    x = np.arange(len(recent))
    slope, _ = np.polyfit(x, recent, 1)
    return slope