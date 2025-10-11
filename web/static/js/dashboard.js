function refreshData() {
    loadSystemOverview();
    loadUtilizationChart();
    loadPathChart();
    loadFlowRateChart();
}

function loadSystemOverview() {
    $.get('/api/v1/system/overview', function(data) {
        const statsGrid = document.getElementById('statsGrid');
        if (!statsGrid) return;
        
        statsGrid.innerHTML = '';
        
        // Topology stats
        if (data.topology) {
            const topoCard = createStatCard('Network Topology', [
                `Switches: ${data.topology.switch_count || 0}`,
                `Links: ${data.topology.link_count || 0}`,
                `Hosts: ${data.topology.host_count || 0}`
            ]);
            statsGrid.appendChild(topoCard);
        }
        
        // Network stats
        if (data.network_stats) {
            const netCard = createStatCard('Network Monitoring', [
                `Monitored Switches: ${data.network_stats.monitored_switches || 0}`,
                `Path Stats: ${data.network_stats.path_stats_count || 0}`
            ]);
            statsGrid.appendChild(netCard);
        }
        
        // Flow monitoring
        if (data.flow_monitoring) {
            const flowCard = createStatCard('Flow Monitoring', [
                `Tracked Flows: ${data.flow_monitoring.tracked_flows || 0}`,
                `Active Measurements: ${data.flow_monitoring.active_measurements || 0}`
            ]);
            statsGrid.appendChild(flowCard);
        }
        
        // Agent status
        if (data.agents) {
            const agentCard = createStatCard('RL Agents', [
                `Agent1: ${data.agents.agent1_enabled ? 'Enabled' : 'Disabled'}`,
                `Baseline Mode: ${data.agents.baseline_mode ? 'On' : 'Off'}`,
                `MPTCP Flows: ${data.agents.mptcp_flows || 0}`
            ]);
            statsGrid.appendChild(agentCard);
        }
        
        // Module status
        if (data.modules) {
            const moduleCard = createStatCard('System Modules', [
                `Topology Manager: ${data.modules.topology_manager ? '✓' : '✗'}`,
                `Network Monitor: ${data.modules.network_monitor ? '✓' : '✗'}`,
                `Flow Rate Monitor: ${data.modules.flow_rate_monitor ? '✓' : '✗'}`,
                `Delay Detector: ${data.modules.delay_detector ? '✓' : '✗'}`
            ]);
            statsGrid.appendChild(moduleCard);
        }
        
    }).fail(function() {
        console.error('Failed to load system overview');
        const statsGrid = document.getElementById('statsGrid');
        if (statsGrid) {
            statsGrid.innerHTML = '<p>Failed to load system overview</p>';
        }
    });
}

function createStatCard(title, stats) {
    const card = document.createElement('div');
    card.className = 'stat-card';
    
    const cardTitle = document.createElement('h3');
    cardTitle.textContent = title;
    card.appendChild(cardTitle);
    
    const statsList = document.createElement('ul');
    stats.forEach(stat => {
        const listItem = document.createElement('li');
        listItem.textContent = stat;
        statsList.appendChild(listItem);
    });
    card.appendChild(statsList);
    
    return card;
}

function loadUtilizationChart() {
    $.get('/api/v1/stats/utilization', function(data) {
        const chartDiv = document.getElementById('utilizationChart');
        if (!chartDiv) return;
        
        if (!data.port_utilization || Object.keys(data.port_utilization).length === 0) {
            chartDiv.innerHTML = '<p style="text-align: center; padding: 50px;">No utilization data available</p>';
            return;
        }
        
        const traces = [];
        const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'];
        let colorIndex = 0;
        
        for (const [switchId, ports] of Object.entries(data.port_utilization)) {
            if (typeof ports === 'object' && ports !== null) {
                for (const [portNo, portData] of Object.entries(ports)) {
                    const utilization = typeof portData === 'object' ? 
                        portData.utilization || 0 : portData || 0;
                    
                    traces.push({
                        x: [switchId],
                        y: [utilization],
                        type: 'bar',
                        name: `S${switchId} Port ${portNo}`,
                        marker: { color: colors[colorIndex % colors.length] }
                    });
                    colorIndex++;
                }
            }
        }
        
        const layout = {
            title: 'Port Utilization',
            xaxis: { title: 'Switch' },
            yaxis: { title: 'Utilization (%)' },
            showlegend: true
        };
        
        Plotly.newPlot(chartDiv, traces, layout);
        
    }).fail(function() {
        const chartDiv = document.getElementById('utilizationChart');
        if (chartDiv) {
            chartDiv.innerHTML = '<p style="text-align: center; padding: 50px;">Failed to load utilization data</p>';
        }
    });
}

function loadPathChart() {
    $.get('/api/v1/stats/paths', function(data) {
        const chartDiv = document.getElementById('pathChart');
        if (!chartDiv) return;
        
        if (!data.path_stats || data.path_stats.length === 0) {
            chartDiv.innerHTML = '<p style="text-align: center; padding: 50px;">No path data available</p>';
            return;
        }
        
        const pathIds = data.path_stats.map(p => p.path_id || 'Unknown');
        const pathCounts = data.path_stats.map(p => p.packet_count || 0);
        
        const trace = {
            x: pathIds,
            y: pathCounts,
            type: 'bar',
            marker: { color: '#2ca02c' }
        };
        
        const layout = {
            title: 'Path Statistics',
            xaxis: { title: 'Path ID' },
            yaxis: { title: 'Packet Count' }
        };
        
        Plotly.newPlot(chartDiv, [trace], layout);
        
    }).fail(function() {
        const chartDiv = document.getElementById('pathChart');
        if (chartDiv) {
            chartDiv.innerHTML = '<p style="text-align: center; padding: 50px;">Failed to load path data</p>';
        }
    });
}

function loadFlowRateChart() {
    $.get('/api/v1/flows/rates', function(data) {
        const chartDiv = document.getElementById('flowRateChart');
        if (!chartDiv) return;
        
        if (!data.flow_rates || data.flow_rates.length === 0) {
            chartDiv.innerHTML = '<p style="text-align: center; padding: 50px;">No flow rate data available</p>';
            return;
        }
        
        const flowIds = data.flow_rates.map(f => f.flow_id || 'Unknown');
        const rates = data.flow_rates.map(f => f.packets_per_sec || 0);
        
        const trace = {
            x: flowIds,
            y: rates,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#e74c3c' },
            name: 'Flow Rate'
        };
        
        const layout = {
            title: 'Flow Rates',
            xaxis: { title: 'Flow ID' },
            yaxis: { title: 'Packets/sec' }
        };
        
        Plotly.newPlot(chartDiv, [trace], layout);
        
    }).fail(function() {
        const chartDiv = document.getElementById('flowRateChart');
        if (chartDiv) {
            chartDiv.innerHTML = '<p style="text-align: center; padding: 50px;">Failed to load flow rate data</p>';
        }
    });
}

// Auto-refresh every 30 seconds
setInterval(refreshData, 30000);

// Initial load
refreshData();