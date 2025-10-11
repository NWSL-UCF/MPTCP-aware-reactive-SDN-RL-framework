#!/usr/bin/env python3
"""
live_metrics_plot.py
────────────────────
Watch the newest JSON-Lines file in ./data/Metrics/ and plot every numeric
field in real time. One subplot per metric, auto-scaling as data arrives.
Groups related metrics (like stddev metrics) in the same subplot.
"""
import json
import time
import queue
import threading
from pathlib import Path
from typing import Union

from flask import Flask, Response, stream_with_context, jsonify

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
PORT = 9999
BASE_DIR = Path(__file__).parent          # folder that contains this script
METRICS_DIR = BASE_DIR / "Metrics"        # where your *.jsonl files live
REFRESH_MS = 250                           # redraw interval (ms)
SLEEP_SEC = 0.3                           # file-poll back-off when idle

# ----------------------------------------------------------------------
# Flask app and event queue
# ----------------------------------------------------------------------
app = Flask(__name__)
event_queue = queue.Queue(maxsize=10_000)  # type: queue.Queue[str]
refresh_queue = queue.Queue(maxsize=10)    # type: queue.Queue[str]

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def _get_latest_metrics_file(metrics_dir: Union[str, Path]) -> Path:
    """Return Path to the newest *.json in `metrics_dir`."""
    metrics_dir = Path(metrics_dir)
    if not metrics_dir.is_dir():
        raise FileNotFoundError(f"{metrics_dir} does not exist")

    jsonl_files = list(metrics_dir.glob("*.json"))
    if not jsonl_files:
        raise FileNotFoundError(f"No *.json files in {metrics_dir}")

    return max(jsonl_files, key=lambda p: p.stat().st_mtime)

# ----------------------------------------------------------------------
# Background thread: watch for newest file and tail it
# ----------------------------------------------------------------------
def _file_watcher():
    """Watch for the newest metrics file and tail it in real-time."""
    current_file = None
    fp = None
    
    while True:
        try:
            # Check for refresh signal
            force_refresh = False
            try:
                refresh_queue.get_nowait()
                force_refresh = True
                # Clear any additional refresh requests
                while not refresh_queue.empty():
                    refresh_queue.get_nowait()
            except queue.Empty:
                pass
            
            # Check for the latest file
            latest = _get_latest_metrics_file(METRICS_DIR)
            
            # If we're tracking a different file or force refresh, switch to the new one
            if latest != current_file or force_refresh:
                if fp:
                    fp.close()
                
                current_file = latest
                print(f"[watcher] tailing {latest}")
                
                # Send a refresh signal to clear existing data
                if force_refresh:
                    event_queue.put(json.dumps({"_refresh": True}))
                
                # Open the new file and read existing content
                fp = open(latest, 'r')
                
                # 1) Read all existing complete lines (history)
                for line in fp:
                    line = line.rstrip("\n")
                    if line:  # Skip empty lines
                        try:
                            # Validate JSON before queuing
                            json.loads(line)
                            event_queue.put(line)
                        except json.JSONDecodeError:
                            print(f"[watcher] skipping invalid JSON: {line}")
                
                # 2) Position at EOF for real-time tailing
                fp.seek(0, 2)
            
            # Check if an even newer file has appeared
            newest = _get_latest_metrics_file(METRICS_DIR)
            if newest != current_file:
                continue  # Loop will handle the switch
            
            # Read new lines
            pos = fp.tell()
            line = fp.readline()
            
            if not line:
                # No new data, sleep and continue
                time.sleep(SLEEP_SEC)
                continue
                
            if not line.endswith("\n"):
                # Partial line, reset position and wait
                fp.seek(pos)
                time.sleep(SLEEP_SEC)
                continue
            
            # Complete line found
            line = line.rstrip("\n")
            if line:  # Skip empty lines
                try:
                    # Validate JSON before queuing
                    json.loads(line)
                    event_queue.put(line)
                except json.JSONDecodeError:
                    print(f"[watcher] skipping invalid JSON: {line}")

        except FileNotFoundError:
            print(f"[watcher] waiting for metrics files in {METRICS_DIR}")
            if fp:
                fp.close()
                fp = None
            current_file = None
            time.sleep(2)
            
        except Exception as exc:
            print(f"[watcher] error: {exc}")
            if fp:
                fp.close()
                fp = None
            current_file = None
            time.sleep(2)

# ----------------------------------------------------------------------
# Flask routes
# ----------------------------------------------------------------------
@app.route("/")
def index():
    """Single-page dashboard – Plotly + JS that consumes /stream."""
    return """
<!doctype html>
<html>
<head>
    <title>Live Metrics</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        :root {
            --bg-primary: #f5f5f5;
            --bg-secondary: white;
            --text-primary: #333;
            --text-secondary: #666;
            --border-color: rgba(0,0,0,0.1);
            --shadow-color: rgba(0,0,0,0.1);
            --grid-color: #e0e0e0;
            --btn-bg: #1976d2;
            --btn-hover: #1565c0;
            --success-color: #388e3c;
            --error-color: #d32f2f;
        }
        
        body.dark-mode {
            --bg-primary: #121212;
            --bg-secondary: #1e1e1e;
            --text-primary: #e0e0e0;
            --text-secondary: #b0b0b0;
            --border-color: rgba(255,255,255,0.1);
            --shadow-color: rgba(0,0,0,0.5);
            --grid-color: #333;
            --btn-bg: #90caf9;
            --btn-hover: #64b5f6;
            --success-color: #81c784;
            --error-color: #f44336;
        }
        
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            transition: background-color 0.3s, color 0.3s;
        }
        
        h2 {
            color: var(--text-primary);
            margin-bottom: 10px;
            display: inline-block;
        }
        
        .controls {
            float: right;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        #main-plot {
            background-color: var(--bg-secondary);
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px var(--shadow-color);
            height: calc(100vh - 120px);
            min-height: 600px;
            transition: background-color 0.3s;
        }
        
        #status {
            margin: 10px 0;
            padding: 10px;
            background-color: var(--bg-secondary);
            border-radius: 5px;
            color: var(--text-secondary);
            font-size: 14px;
            display: inline-block;
            transition: background-color 0.3s, color 0.3s;
        }
        
        .error {
            color: var(--error-color) !important;
        }
        
        .connected {
            color: var(--success-color) !important;
        }
        
        .btn {
            background-color: var(--btn-bg);
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: background-color 0.3s;
        }
        
        .btn:hover {
            background-color: var(--btn-hover);
        }
        
        .btn:active {
            transform: scale(0.98);
        }
        
        /* Dark mode toggle */
        .theme-switch {
            position: relative;
            display: inline-block;
            width: 50px;
            height: 24px;
        }
        
        .theme-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        
        .slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            transition: .4s;
            border-radius: 24px;
        }
        
        .slider:before {
            position: absolute;
            content: "☀️";
            height: 18px;
            width: 18px;
            left: 3px;
            bottom: 3px;
            transition: .4s;
            font-size: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        input:checked + .slider {
            background-color: #2196F3;
        }
        
        input:checked + .slider:before {
            content: "🌙";
            transform: translateX(26px);
        }
        
        .theme-label {
            margin-right: 8px;
            font-size: 14px;
            color: var(--text-secondary);
        }
        
        .clearfix::after {
            content: "";
            display: table;
            clear: both;
        }
    </style>
</head>
<body>
    <div class="clearfix">
        <h2>SDN-RL Metrics (live)</h2>
        <div class="controls">
            <span class="theme-label">Theme:</span>
            <label class="theme-switch">
                <input type="checkbox" id="darkModeToggle">
                <span class="slider"></span>
            </label>
            <button class="btn" id="refreshBtn" onclick="refreshData()">🔄 Refresh Latest File</button>
            <a href="/links" class="btn">📊 Link Utilization View</a>
        </div>
    </div>
    <div id="status">Connecting...</div>
    <div id="main-plot"></div>
    
    <script>
    // Initialize
    const plotDiv = document.getElementById('main-plot');
    const statusDiv = document.getElementById('status');
    const darkModeToggle = document.getElementById('darkModeToggle');
    const refreshBtn = document.getElementById('refreshBtn');
    
    // Track metrics - both individual and grouped
    let metrics = {};
    let metricGroups = {}; // For grouped metrics like stddev
    let plotNames = []; // Names of plots (individual metrics + groups)
    let dataPoints = 0;
    let maxPoints = 500;
    let plotInitialized = false;
    
    // Color palette - expanded for switch lines and grouped metrics
    const lightColors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ];
    
    const darkColors = [
        '#64b5f6', '#ffb74d', '#81c784', '#e57373', '#ba68c8',
        '#a1887f', '#f06292', '#b0b0b0', '#dce775', '#4dd0e1'
    ];
    
    const lightSwitchColors = [
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
        '#ffff33', '#a65628', '#f781bf', '#999999'
    ];
    
    const darkSwitchColors = [
        '#ef5350', '#42a5f5', '#66bb6a', '#ab47bc', '#ffa726',
        '#ffee58', '#8d6e63', '#ec407a', '#bdbdbd'
    ];
    
    let colors = lightColors;
    let switchColors = lightSwitchColors;
    
    // Dark mode functionality
    function loadDarkMode() {
        const isDark = localStorage.getItem('darkMode') === 'true';
        darkModeToggle.checked = isDark;
        applyDarkMode(isDark);
    }
    
    function applyDarkMode(isDark) {
        document.body.classList.toggle('dark-mode', isDark);
        colors = isDark ? darkColors : lightColors;
        switchColors = isDark ? darkSwitchColors : lightSwitchColors;
        
        // Update Plotly theme
        if (plotInitialized) {
            const plotBg = isDark ? '#1e1e1e' : 'white';
            const gridColor = isDark ? '#333' : '#e0e0e0';
            const textColor = isDark ? '#e0e0e0' : '#333';
            
            const update = {
                plot_bgcolor: plotBg,
                paper_bgcolor: plotBg,
                font: { color: textColor }
            };
            
            // Update grid colors for all axes
            plotNames.forEach((_, idx) => {
                update[`xaxis${idx + 1}.gridcolor`] = gridColor;
                update[`yaxis${idx + 1}.gridcolor`] = gridColor;
                update[`xaxis${idx + 1}.color`] = textColor;
                update[`yaxis${idx + 1}.color`] = textColor;
            });
            
            Plotly.relayout(plotDiv, update);
            recreateSubplots(); // Recreate to update trace colors
        }
    }
    
    darkModeToggle.addEventListener('change', (e) => {
        const isDark = e.target.checked;
        localStorage.setItem('darkMode', isDark);
        applyDarkMode(isDark);
    });
    
    // Refresh functionality
    function refreshData() {
        refreshBtn.disabled = true;
        refreshBtn.textContent = '⏳ Refreshing...';
        
        // Clear existing data
        metrics = {};
        metricGroups = {};
        plotNames = [];
        dataPoints = 0;
        plotInitialized = false;
        
        // Clear plot
        Plotly.purge(plotDiv);
        
        // Send refresh signal to backend
        fetch('/refresh', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                statusDiv.textContent = 'Refreshed - loading new data...';
                statusDiv.className = 'connected';
                
                // Re-initialize plot
                const isDark = document.body.classList.contains('dark-mode');
                const plotBg = isDark ? '#1e1e1e' : 'white';
                const textColor = isDark ? '#e0e0e0' : '#333';
                
                const layout = {
                    grid: {
                        rows: 1,
                        columns: 1,
                        pattern: 'independent',
                        roworder: 'top to bottom'
                    },
                    height: window.innerHeight - 140,
                    showlegend: false,
                    margin: { l: 60, r: 30, t: 30, b: 40 },
                    plot_bgcolor: plotBg,
                    paper_bgcolor: plotBg,
                    font: { color: textColor }
                };
                
                Plotly.newPlot(plotDiv, [], layout, {responsive: true});
            })
            .catch(err => {
                console.error('Refresh error:', err);
                statusDiv.textContent = 'Refresh failed';
                statusDiv.className = 'error';
            })
            .finally(() => {
                setTimeout(() => {
                    refreshBtn.disabled = false;
                    refreshBtn.textContent = '🔄 Refresh Latest File';
                }, 1000);
            });
    }
    
    // Define metric groups - metrics that should be plotted together
    const metricGroupDefs = {
        'stddev_metrics': {
            displayName: 'Standard Deviation Metrics',
            metrics: ['sdlus_stddev', 'link_stddev'],
            colors: ['#d62728', '#2ca02c'], // Red for switch stddev, green for link stddev
            labels: ['Switch MLU StdDev', 'Link Utilization StdDev']
        }
    };
    
    // Initialize empty plot
    loadDarkMode();
    const isDark = document.body.classList.contains('dark-mode');
    const plotBg = isDark ? '#1e1e1e' : 'white';
    const textColor = isDark ? '#e0e0e0' : '#333';
    
    const layout = {
        grid: {
            rows: 1,
            columns: 1,
            pattern: 'independent',
            roworder: 'top to bottom'
        },
        height: window.innerHeight - 140,
        showlegend: false,
        margin: { l: 60, r: 30, t: 30, b: 40 },
        plot_bgcolor: plotBg,
        paper_bgcolor: plotBg,
        font: { color: textColor }
    };
    
    Plotly.newPlot(plotDiv, [], layout, {responsive: true});
    
    // Function to determine if a metric belongs to a group
    function getMetricGroup(metricName) {
        for (const [groupName, groupDef] of Object.entries(metricGroupDefs)) {
            if (groupDef.metrics.includes(metricName)) {
                return groupName;
            }
        }
        return null;
    }
    
    // Function to recreate plot with new metrics
    function recreateSubplots() {
        const numPlots = plotNames.length;
        if (numPlots === 0) return;
        
        // Calculate grid layout
        const cols = Math.min(2, numPlots);
        const rows = Math.ceil(numPlots / cols);
        
        // Create traces
        const traces = [];
        const annotations = [];
        
        plotNames.forEach((plotName, plotIdx) => {
            const row = Math.floor(plotIdx / cols) + 1;
            const col = (plotIdx % cols) + 1;
            
            if (metricGroups[plotName]) {
                // Handle grouped metrics
                const group = metricGroups[plotName];
                const groupDef = metricGroupDefs[plotName];
                
                group.metricNames.forEach((metricName, metricIdx) => {
                    const metric = group.metrics[metricName];
                    if (!metric) return;
                    
                    if (metric.multiSeries) {
                        // Handle multi-series metrics within group
                        metric.switches.forEach((switchId, switchIdx) => {
                            const currentValue = metric.latest[switchId];
                            const nameWithValue = `${groupDef.labels[metricIdx]} - Switch ${switchId}: ${formatValue(currentValue)}`;
                            
                            traces.push({
                                x: metric.data[switchId].x,
                                y: metric.data[switchId].y,
                                type: 'scatter',
                                mode: 'lines',
                                name: nameWithValue,
                                line: {
                                    color: groupDef.colors[metricIdx],
                                    width: 2,
                                    dash: switchIdx > 0 ? 'dash' : 'solid'
                                },
                                xaxis: `x${plotIdx + 1}`,
                                yaxis: `y${plotIdx + 1}`,
                                showlegend: true,
                                legendgroup: `g${plotIdx}`
                            });
                        });
                    } else {
                        // Handle single-series metrics within group
                        traces.push({
                            x: metric.data.x,
                            y: metric.data.y,
                            type: 'scatter',
                            mode: 'lines',
                            name: groupDef.labels[metricIdx],
                            line: {
                                color: groupDef.colors[metricIdx],
                                width: 2
                            },
                            xaxis: `x${plotIdx + 1}`,
                            yaxis: `y${plotIdx + 1}`,
                            showlegend: true,
                            legendgroup: `g${plotIdx}`
                        });
                    }
                });
                
                // Add title annotation for group (cleaner without individual values)
                annotations.push({
                    text: `<b>${groupDef.displayName}</b>`,
                    x: 0.5,
                    y: 1.05,
                    xref: `x${plotIdx + 1} domain`,
                    yref: `y${plotIdx + 1} domain`,
                    showarrow: false,
                    font: { size: 14 }
                });
                
            } else if (metrics[plotName]) {
                // Handle individual metrics
                const metric = metrics[plotName];
                
                if (metric.multiSeries) {
                    // Handle multi-series metrics (like switch_mlus)
                    metric.switches.forEach((switchId, switchIdx) => {
                        const currentValue = metric.latest[switchId];
                        const nameWithValue = `Switch ${switchId}: ${formatValue(currentValue)}`;
                        
                        traces.push({
                            x: metric.data[switchId].x,
                            y: metric.data[switchId].y,
                            type: 'scatter',
                            mode: 'lines',
                            name: nameWithValue,
                            line: {
                                color: switchColors[switchIdx % switchColors.length],
                                width: 2
                            },
                            xaxis: `x${plotIdx + 1}`,
                            yaxis: `y${plotIdx + 1}`,
                            showlegend: true,
                            legendgroup: `g${plotIdx}`
                        });
                    });
                    
                    // Add title annotation without values (cleaner)
                    annotations.push({
                        text: `<b>${metric.displayName || plotName}</b>`,
                        x: 0.5,
                        y: 1.05,
                        xref: `x${plotIdx + 1} domain`,
                        yref: `y${plotIdx + 1} domain`,
                        showarrow: false,
                        font: { size: 14 }
                    });
                } else {
                    // Handle single-series metrics
                    traces.push({
                        x: metric.data.x,
                        y: metric.data.y,
                        type: 'scatter',
                        mode: 'lines',
                        name: metric.displayName || plotName,
                        line: {
                            color: colors[plotIdx % colors.length],
                            width: 2
                        },
                        xaxis: `x${plotIdx + 1}`,
                        yaxis: `y${plotIdx + 1}`,
                        showlegend: false
                    });
                    
                    // Add title annotation
                    annotations.push({
                        text: `<b>${metric.displayName || plotName}</b><br>Latest: ${formatValue(metric.latest)}`,
                        x: 0.5,
                        y: 1.05,
                        xref: `x${plotIdx + 1} domain`,
                        yref: `y${plotIdx + 1} domain`,
                        showarrow: false,
                        font: { size: 14 }
                    });
                }
            }
        });
        
        // Get current theme settings
        const isDark = document.body.classList.contains('dark-mode');
        const plotBg = isDark ? '#1e1e1e' : 'white';
        const gridColor = isDark ? '#333' : '#e0e0e0';
        const textColor = isDark ? '#e0e0e0' : '#333';
        
        // Create layout with subplots
        const newLayout = {
            height: window.innerHeight - 140,
            showlegend: true,
            margin: { l: 60, r: 30, t: 40, b: 40 },
            grid: {
                rows: rows,
                columns: cols,
                pattern: 'independent',
                roworder: 'top to bottom',
                xgap: 0.1,
                ygap: 0.15
            },
            annotations: annotations,
            legend: {
                font: { size: 10 },
                bgcolor: 'rgba(255,255,255,0.9)',
                bordercolor: '#333',
                borderwidth: 1,
                orientation: 'v',
                tracegroupgap: 5
            },
            plot_bgcolor: plotBg,
            paper_bgcolor: plotBg,
            font: { color: textColor }
        };
        
        // Add axis configurations
        plotNames.forEach((plotName, idx) => {
            newLayout[`xaxis${idx + 1}`] = {
                type: 'date',
                tickformat: '%H:%M:%S',
                showgrid: true,
                gridcolor: gridColor,
                color: textColor
            };
            newLayout[`yaxis${idx + 1}`] = {
                autorange: true,
                showgrid: true,
                gridcolor: gridColor,
                title: idx % cols === 0 ? 'Value' : '',
                color: textColor
            };
        });
        
        // Redraw entire plot
        Plotly.react(plotDiv, traces, newLayout, {responsive: true});
        plotInitialized = true;
    }
    
    // Function to update all plots
    function updateAllPlots() {
        if (!plotInitialized || plotNames.length === 0) return;
        
        const traces = [];
        const annotations = [];
        
        plotNames.forEach((plotName, plotIdx) => {
            if (metricGroups[plotName]) {
                // Update grouped metrics
                const group = metricGroups[plotName];
                const groupDef = metricGroupDefs[plotName];
                
                group.metricNames.forEach((metricName) => {
                    const metric = group.metrics[metricName];
                    if (!metric) return;
                    
                    if (metric.multiSeries) {
                        metric.switches.forEach((switchId) => {
                            const currentValue = metric.latest[switchId];
                            const nameWithValue = `${groupDef.labels[metricIdx]} - Switch ${switchId}: ${formatValue(currentValue)}`;
                            
                            traces.push({
                                x: metric.data[switchId].x,
                                y: metric.data[switchId].y,
                                name: nameWithValue
                            });
                        });
                    } else {
                        traces.push({
                            x: metric.data.x,
                            y: metric.data.y
                        });
                    }
                });
                
                // Update annotation for group (cleaner)
                annotations.push({
                    text: `<b>${groupDef.displayName}</b>`,
                    x: 0.5,
                    y: 1.05,
                    xref: `x${plotIdx + 1} domain`,
                    yref: `y${plotIdx + 1} domain`,
                    showarrow: false,
                    font: { size: 14 }
                });
                
            } else if (metrics[plotName]) {
                // Update individual metrics
                const metric = metrics[plotName];
                
                if (metric.multiSeries) {
                    // Update multi-series metrics
                    metric.switches.forEach((switchId) => {
                        const currentValue = metric.latest[switchId];
                        const nameWithValue = `Switch ${switchId}: ${formatValue(currentValue)}`;
                        
                        traces.push({
                            x: metric.data[switchId].x,
                            y: metric.data[switchId].y,
                            name: nameWithValue
                        });
                    });
                    
                    // Update annotation without values (cleaner)
                    annotations.push({
                        text: `<b>${metric.displayName || plotName}</b>`,
                        x: 0.5,
                        y: 1.05,
                        xref: `x${plotIdx + 1} domain`,
                        yref: `y${plotIdx + 1} domain`,
                        showarrow: false,
                        font: { size: 14 }
                    });
                } else {
                    // Update single-series metrics
                    traces.push({
                        x: metric.data.x,
                        y: metric.data.y
                    });
                    
                    annotations.push({
                        text: `<b>${metric.displayName || plotName}</b><br>Latest: ${formatValue(metric.latest)}`,
                        x: 0.5,
                        y: 1.05,
                        xref: `x${plotIdx + 1} domain`,
                        yref: `y${plotIdx + 1} domain`,
                        showarrow: false,
                        font: { size: 14 }
                    });
                }
            }
        });
        
        // Update data and annotations
        Plotly.update(plotDiv, traces, { annotations: annotations });
    }
    
    // Connect to SSE stream
    const source = new EventSource("/stream");
    
    source.onopen = () => {
        statusDiv.textContent = 'Connected - waiting for data...';
        statusDiv.className = 'connected';
    };
    
    source.onerror = (e) => {
        statusDiv.textContent = 'Connection error - retrying...';
        statusDiv.className = 'error';
    };
    
    // Helper to format values for display
    function formatValue(value) {
        if (value === null || value === undefined) return 'N/A';
        
        // For very small numbers (scientific notation)
        if (Math.abs(value) < 0.001 && value !== 0) {
            return value.toExponential(3);
        }
        // For regular numbers
        return value.toFixed(4);
    }
    
    // Helper to get display name for metrics
    function getDisplayName(metricKey) {
        const displayNames = {
            'sdlus_stddev': 'Switch MLU StdDev',
            'link_stddev': 'Link Utilization StdDev',
            'max_rr': 'Max Round-Robin',
            // Add more custom names here as needed
        };
        return displayNames[metricKey] || metricKey;
    }
    
    source.onmessage = (evt) => {
        try {
            const obj = JSON.parse(evt.data);
            
            // Handle refresh signal
            if (obj._refresh) {
                metrics = {};
                metricGroups = {};
                plotNames = [];
                dataPoints = 0;
                plotInitialized = false;
                Plotly.purge(plotDiv);
                
                const isDark = document.body.classList.contains('dark-mode');
                const plotBg = isDark ? '#1e1e1e' : 'white';
                const textColor = isDark ? '#e0e0e0' : '#333';
                
                const layout = {
                    grid: {
                        rows: 1,
                        columns: 1,
                        pattern: 'independent',
                        roworder: 'top to bottom'
                    },
                    height: window.innerHeight - 140,
                    showlegend: false,
                    margin: { l: 60, r: 30, t: 30, b: 40 },
                    plot_bgcolor: plotBg,
                    paper_bgcolor: plotBg,
                    font: { color: textColor }
                };
                
                Plotly.newPlot(plotDiv, [], layout, {responsive: true});
                return;
            }
            
            const timestamp = obj.timestamp || Date.now() / 1000;
            const t = new Date(timestamp * 1000);
            
            dataPoints++;
            let needsRecreate = false;
            
            // Process each field
            Object.entries(obj).forEach(([key, value]) => {
                if (key === 'timestamp') return;
                
                // Handle switch_mlus specially (it's an object with multiple switches)
                if (key === 'switch_mlus' && typeof value === 'object') {
                    const metricKey = 'switch_mlus';
                    
                    // Create metric if new
                    if (!metrics[metricKey]) {
                        metrics[metricKey] = {
                            data: {}, // Will contain data for each switch
                            switches: [], // Track which switches we've seen
                            min: Infinity,
                            max: -Infinity,
                            latest: {},
                            displayName: 'Switch MLUs',
                            multiSeries: true
                        };
                        
                        // Add to plot names if not grouped
                        const groupName = getMetricGroup(metricKey);
                        if (!groupName && !plotNames.includes(metricKey)) {
                            plotNames.push(metricKey);
                            needsRecreate = true;
                        }
                    }
                    
                    const metric = metrics[metricKey];
                    
                    // Process each switch
                    Object.entries(value).forEach(([switchId, switchValue]) => {
                        if (typeof switchValue !== 'number') return;
                        
                        // Initialize data for this switch if new
                        if (!metric.data[switchId]) {
                            metric.data[switchId] = { x: [], y: [] };
                            metric.switches.push(switchId);
                            metric.switches.sort(); // Keep switches in order
                            needsRecreate = true;
                        }
                        
                        // Add new data point
                        metric.data[switchId].x.push(t);
                        metric.data[switchId].y.push(switchValue);
                        metric.latest[switchId] = switchValue;
                        metric.min = Math.min(metric.min, switchValue);
                        metric.max = Math.max(metric.max, switchValue);
                        
                        // Keep only last N points
                        if (metric.data[switchId].x.length > maxPoints) {
                            metric.data[switchId].x.shift();
                            metric.data[switchId].y.shift();
                        }
                    });
                    return;
                }
                
                // Handle regular numeric fields
                if (typeof value !== 'number') return;
                
                // Check if this metric belongs to a group
                const groupName = getMetricGroup(key);
                
                if (groupName) {
                    // Handle grouped metric
                    if (!metricGroups[groupName]) {
                        metricGroups[groupName] = {
                            metrics: {},
                            metricNames: []
                        };
                        plotNames.push(groupName);
                        needsRecreate = true;
                    }
                    
                    const group = metricGroups[groupName];
                    
                    // Create metric if new
                    if (!group.metrics[key]) {
                        group.metrics[key] = {
                            data: { x: [], y: [] },
                            min: Infinity,
                            max: -Infinity,
                            latest: null,
                            displayName: getDisplayName(key),
                            multiSeries: false
                        };
                        group.metricNames.push(key);
                        needsRecreate = true;
                    }
                    
                    const metric = group.metrics[key];
                    
                    // Add new data point
                    metric.data.x.push(t);
                    metric.data.y.push(value);
                    metric.latest = value;
                    metric.min = Math.min(metric.min, value);
                    metric.max = Math.max(metric.max, value);
                    
                    // Keep only last N points
                    if (metric.data.x.length > maxPoints) {
                        metric.data.x.shift();
                        metric.data.y.shift();
                    }
                } else {
                    // Handle individual metric
                    if (!metrics[key]) {
                        metrics[key] = {
                            data: { x: [], y: [] },
                            min: Infinity,
                            max: -Infinity,
                            latest: null,
                            displayName: getDisplayName(key),
                            multiSeries: false
                        };
                        plotNames.push(key);
                        needsRecreate = true;
                    }
                    
                    const metric = metrics[key];
                    
                    // Add new data point
                    metric.data.x.push(t);
                    metric.data.y.push(value);
                    metric.latest = value;
                    metric.min = Math.min(metric.min, value);
                    metric.max = Math.max(metric.max, value);
                    
                    // Keep only last N points
                    if (metric.data.x.length > maxPoints) {
                        metric.data.x.shift();
                        metric.data.y.shift();
                    }
                }
            });
            
            // Update status
            statusDiv.textContent = `Connected - ${dataPoints} data points, ${plotNames.length} plots`;
            statusDiv.className = 'connected';
            
            // Update plots
            if (needsRecreate) {
                recreateSubplots();
            } else {
                updateAllPlots();
            }
            
        } catch (e) {
            console.error('Error processing message:', e);
        }
    };
    
    // Handle window resize
    window.addEventListener('resize', () => {
        if (plotInitialized) {
            Plotly.relayout(plotDiv, {
                height: window.innerHeight - 140
            });
        }
    });
    </script>
</body>
</html>"""

@app.route("/links")
def links_view():
    """Dedicated Link Utilization view with plot on left and metrics on right."""
    return """
<!doctype html>
<html>
<head>
    <title>Link Utilization Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        :root {
            --bg-primary: #f5f5f5;
            --bg-secondary: white;
            --text-primary: #333;
            --text-secondary: #666;
            --border-color: rgba(0,0,0,0.1);
            --shadow-color: rgba(0,0,0,0.1);
            --grid-color: #e0e0e0;
            --btn-bg: #1976d2;
            --btn-hover: #1565c0;
            --success-color: #388e3c;
            --error-color: #d32f2f;
            --util-very-low: #fafafa;
            --util-medium: #4caf50;
            --util-high: #ff9800;
            --util-very-high: #f44336;
        }
        
        body.dark-mode {
            --bg-primary: #121212;
            --bg-secondary: #1e1e1e;
            --text-primary: #e0e0e0;
            --text-secondary: #b0b0b0;
            --border-color: rgba(255,255,255,0.1);
            --shadow-color: rgba(0,0,0,0.5);
            --grid-color: #333;
            --btn-bg: #90caf9;
            --btn-hover: #64b5f6;
            --success-color: #81c784;
            --error-color: #f44336;
            --util-very-low: #2a2a2a;
            --util-medium: #66bb6a;
            --util-high: #ffb74d;
            --util-very-high: #ef5350;
        }
        
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            transition: background-color 0.3s, color 0.3s;
        }
        
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        h2 {
            color: var(--text-primary);
            margin: 0;
        }
        
        .controls {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .container {
            display: flex;
            gap: 20px;
            height: calc(100vh - 120px);
        }
        
        .plot-container {
            flex: 5;
            background-color: var(--bg-secondary);
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px var(--shadow-color);
            transition: background-color 0.3s;
        }
        
        .metrics-panel {
            flex: 1;
            min-width: 240px;
            max-width: 280px;
            background-color: var(--bg-secondary);
            padding: 16px;
            border-radius: 8px;
            box-shadow: 0 2px 8px var(--shadow-color);
            overflow-y: auto;
            transition: background-color 0.3s;
        }
        
        .metrics-header {
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 12px;
            text-align: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 6px;
        }
        
        .metric-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px;
            margin-bottom: 4px;
            border-radius: 4px;
            transition: all 0.3s ease;
            border-left: 3px solid transparent;
            font-size: 13px;
        }
        
        .metric-item.very-high {
            background-color: var(--util-very-high);
            color: white;
            border-left-color: #c62828;
        }
        
        .metric-item.high {
            background-color: var(--util-high);
            color: white;
            border-left-color: #f57c00;
        }
        
        .metric-item.medium {
            background-color: var(--util-medium);
            color: white;
            border-left-color: #388e3c;
        }
        
        .metric-item.low {
            background-color: rgba(76, 175, 80, 0.2);
            color: var(--text-primary);
            border-left-color: rgba(76, 175, 80, 0.6);
        }
        
        body.dark-mode .metric-item.low {
            background-color: rgba(102, 187, 106, 0.2);
            border-left-color: rgba(102, 187, 106, 0.6);
        }
        
        .metric-item.very-low {
            background-color: var(--util-very-low);
            color: var(--text-secondary);
            border: 1px solid var(--border-color);
            border-left-color: #bdbdbd;
        }
        
        .metric-name {
            font-weight: 500;
            font-size: 12px;
        }
        
        .metric-value {
            font-weight: bold;
            font-family: monospace;
            font-size: 12px;
        }
        
        #status {
            margin: 10px 0;
            padding: 10px;
            background-color: var(--bg-secondary);
            border-radius: 5px;
            color: var(--text-secondary);
            font-size: 14px;
            transition: background-color 0.3s, color 0.3s;
        }
        
        .error {
            color: var(--error-color) !important;
        }
        
        .connected {
            color: var(--success-color) !important;
        }
        
        .btn {
            background-color: var(--btn-bg);
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: background-color 0.3s;
            text-decoration: none;
            display: inline-block;
        }
        
        .btn:hover {
            background-color: var(--btn-hover);
        }
        
        /* Dark mode toggle */
        .theme-switch {
            position: relative;
            display: inline-block;
            width: 50px;
            height: 24px;
        }
        
        .theme-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        
        .slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            transition: .4s;
            border-radius: 24px;
        }
        
        .slider:before {
            position: absolute;
            content: "☀️";
            height: 18px;
            width: 18px;
            left: 3px;
            bottom: 3px;
            transition: .4s;
            font-size: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        input:checked + .slider {
            background-color: #2196F3;
        }
        
        input:checked + .slider:before {
            content: "🌙";
            transform: translateX(26px);
        }
        
        .theme-label {
            margin-right: 8px;
            font-size: 14px;
            color: var(--text-secondary);
        }
        
        #plot {
            width: 100%;
            height: 100%;
        }
    </style>
</head>
<body>
    <div class="header">
        <h2>🔗 Link Utilization Dashboard</h2>
        <div class="controls">
            <span class="theme-label">Theme:</span>
            <label class="theme-switch">
                <input type="checkbox" id="darkModeToggle">
                <span class="slider"></span>
            </label>
            <button class="btn" id="refreshBtn" onclick="refreshData()">🔄 Refresh</button>
            <a href="/" class="btn">📊 All Metrics</a>
        </div>
    </div>
    
    <div id="status">Connecting...</div>
    
    <div class="container">
        <div class="plot-container">
            <div id="plot"></div>
        </div>
        
        <div class="metrics-panel">
            <div class="metrics-header">Link Utilization Status</div>
            <div id="metrics-list">
                <div style="text-align: center; color: var(--text-secondary); padding: 20px;">
                    Waiting for link data...
                </div>
            </div>
        </div>
    </div>
    
    <script>
    // Initialize
    const plotDiv = document.getElementById('plot');
    const statusDiv = document.getElementById('status');
    const metricsListDiv = document.getElementById('metrics-list');
    const darkModeToggle = document.getElementById('darkModeToggle');
    const refreshBtn = document.getElementById('refreshBtn');
    
    let linkMetrics = {};
    let dataPoints = 0;
    let maxPoints = 500;
    let plotInitialized = false;
    
    // Link colors
    const lightLinkColors = [
        '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3',
        '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd',
        '#ccebc5', '#ffed6f'
    ];
    
    const darkLinkColors = [
        '#4db6ac', '#fff176', '#9575cd', '#f06292', '#64b5f6',
        '#ffb74d', '#aed581', '#f8bbd9', '#e0e0e0', '#ce93d8',
        '#c8e6c9', '#fff59d'
    ];
    
    let linkColors = lightLinkColors;
    
    // Dark mode functionality
    function loadDarkMode() {
        const isDark = localStorage.getItem('darkMode') === 'true';
        darkModeToggle.checked = isDark;
        applyDarkMode(isDark);
    }
    
    function applyDarkMode(isDark) {
        document.body.classList.toggle('dark-mode', isDark);
        linkColors = isDark ? darkLinkColors : lightLinkColors;
        
        if (plotInitialized) {
            updatePlot();
        }
    }
    
    darkModeToggle.addEventListener('change', (e) => {
        const isDark = e.target.checked;
        localStorage.setItem('darkMode', isDark);
        applyDarkMode(isDark);
    });
    
    // Refresh functionality
    function refreshData() {
        refreshBtn.disabled = true;
        refreshBtn.textContent = '⏳ Refreshing...';
        
        linkMetrics = {};
        dataPoints = 0;
        plotInitialized = false;
        
        Plotly.purge(plotDiv);
        metricsListDiv.innerHTML = '<div style="text-align: center; color: var(--text-secondary); padding: 20px;">Waiting for link data...</div>';
        
        fetch('/refresh', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                statusDiv.textContent = 'Refreshed - loading new data...';
                statusDiv.className = 'connected';
                initializePlot();
            })
            .catch(err => {
                console.error('Refresh error:', err);
                statusDiv.textContent = 'Refresh failed';
                statusDiv.className = 'error';
            })
            .finally(() => {
                setTimeout(() => {
                    refreshBtn.disabled = false;
                    refreshBtn.textContent = '🔄 Refresh';
                }, 1000);
            });
    }
    
    // Color classification function
    function getUtilizationClass(value) {
        if (value >= 0.9) return 'very-high';   // 90-100%
        if (value >= 0.75) return 'high';       // 75-90%
        if (value >= 0.3) return 'medium';      // 30-75%
        if (value >= 0.05) return 'low';        // 5-30%
        return 'very-low';                      // 0-5%
    }
    
    // Format value for display
    function formatValue(value) {
        if (value === null || value === undefined) return 'N/A';
        return (value * 100).toFixed(1) + '%';
    }
    
    // Initialize plot
    function initializePlot() {
        const isDark = document.body.classList.contains('dark-mode');
        const plotBg = isDark ? '#1e1e1e' : 'white';
        const gridColor = isDark ? '#333' : '#e0e0e0';
        const textColor = isDark ? '#e0e0e0' : '#333';
        
        const layout = {
            title: {
                text: 'Link Utilization Over Time',
                font: { size: 18, color: textColor }
            },
            xaxis: {
                type: 'date',
                tickformat: '%H:%M:%S',
                showgrid: true,
                gridcolor: gridColor,
                color: textColor,
                title: 'Time'
            },
            yaxis: {
                title: 'Utilization (%)',
                showgrid: true,
                gridcolor: gridColor,
                color: textColor,
                range: [0, 100]
            },
            plot_bgcolor: plotBg,
            paper_bgcolor: plotBg,
            font: { color: textColor },
            margin: { l: 60, r: 30, t: 60, b: 50 },
            showlegend: true,
            legend: {
                orientation: 'v',
                x: 1.01,
                y: 1,
                font: { size: 9 }
            }
        };
        
        Plotly.newPlot(plotDiv, [], layout, {responsive: true});
        plotInitialized = true;
    }
    
    // Update plot with current data
    function updatePlot() {
        if (!plotInitialized || Object.keys(linkMetrics).length === 0) return;
        
        const traces = [];
        const linkIds = Object.keys(linkMetrics).sort();
        
        linkIds.forEach((linkId, idx) => {
            const metric = linkMetrics[linkId];
            traces.push({
                x: metric.data.x,
                y: metric.data.y.map(v => v * 100), // Convert to percentage
                type: 'scatter',
                mode: 'lines',
                name: `Link ${linkId}`,
                line: {
                    color: linkColors[idx % linkColors.length],
                    width: 2
                }
            });
        });
        
        const isDark = document.body.classList.contains('dark-mode');
        const plotBg = isDark ? '#1e1e1e' : 'white';
        const gridColor = isDark ? '#333' : '#e0e0e0';
        const textColor = isDark ? '#e0e0e0' : '#333';
        
        const layout = {
            title: {
                text: 'Link Utilization Over Time',
                font: { size: 18, color: textColor }
            },
            xaxis: {
                type: 'date',
                tickformat: '%H:%M:%S',
                showgrid: true,
                gridcolor: gridColor,
                color: textColor,
                title: 'Time'
            },
            yaxis: {
                title: 'Utilization (%)',
                showgrid: true,
                gridcolor: gridColor,
                color: textColor,
                range: [0, 100]
            },
            plot_bgcolor: plotBg,
            paper_bgcolor: plotBg,
            font: { color: textColor },
            margin: { l: 60, r: 30, t: 60, b: 50 },
            showlegend: true,
            legend: {
                orientation: 'v',
                x: 1.01,
                y: 1,
                font: { size: 9 }
            }
        };
        
        Plotly.react(plotDiv, traces, layout, {responsive: true});
    }
    
    // Update metrics panel
    function updateMetricsPanel() {
        if (Object.keys(linkMetrics).length === 0) {
            metricsListDiv.innerHTML = '<div style="text-align: center; color: var(--text-secondary); padding: 20px;">Waiting for link data...</div>';
            return;
        }
        
        // Sort links by utilization (descending)
        const sortedLinks = Object.entries(linkMetrics)
            .map(([linkId, metric]) => ({
                id: linkId,
                value: metric.latest || 0
            }))
            .sort((a, b) => b.value - a.value);
        
        const html = sortedLinks.map(({id, value}) => {
            const className = getUtilizationClass(value);
            return `
                <div class="metric-item ${className}">
                    <span class="metric-name">Link ${id}</span>
                    <span class="metric-value">${formatValue(value)}</span>
                </div>
            `;
        }).join('');
        
        metricsListDiv.innerHTML = html;
    }
    
    // Initialize
    loadDarkMode();
    initializePlot();
    
    // Connect to SSE stream
    const source = new EventSource("/stream");
    
    source.onopen = () => {
        statusDiv.textContent = 'Connected - waiting for data...';
        statusDiv.className = 'connected';
    };
    
    source.onerror = (e) => {
        statusDiv.textContent = 'Connection error - retrying...';
        statusDiv.className = 'error';
    };
    
    source.onmessage = (evt) => {
        try {
            const obj = JSON.parse(evt.data);
            
            // Handle refresh signal
            if (obj._refresh) {
                linkMetrics = {};
                dataPoints = 0;
                plotInitialized = false;
                Plotly.purge(plotDiv);
                metricsListDiv.innerHTML = '<div style="text-align: center; color: var(--text-secondary); padding: 20px;">Waiting for link data...</div>';
                initializePlot();
                return;
            }
            
            // Only process Links_utilizations data
            if (obj.Links_utilizations && typeof obj.Links_utilizations === 'object') {
                const timestamp = obj.timestamp || Date.now() / 1000;
                const t = new Date(timestamp * 1000);
                
                dataPoints++;
                let needsUpdate = false;
                
                Object.entries(obj.Links_utilizations).forEach(([linkId, utilization]) => {
                    if (typeof utilization !== 'number') return;
                    
                    // Initialize link metric if new
                    if (!linkMetrics[linkId]) {
                        linkMetrics[linkId] = {
                            data: { x: [], y: [] },
                            latest: 0
                        };
                        needsUpdate = true;
                    }
                    
                    const metric = linkMetrics[linkId];
                    
                    // Add new data point
                    metric.data.x.push(t);
                    metric.data.y.push(utilization);
                    metric.latest = utilization;
                    
                    // Keep only last N points
                    if (metric.data.x.length > maxPoints) {
                        metric.data.x.shift();
                        metric.data.y.shift();
                    }
                    
                    needsUpdate = true;
                });
                
                if (needsUpdate) {
                    updatePlot();
                    updateMetricsPanel();
                    
                    const linkCount = Object.keys(linkMetrics).length;
                    statusDiv.textContent = `Connected - ${dataPoints} data points, ${linkCount} links`;
                    statusDiv.className = 'connected';
                }
            }
            
        } catch (e) {
            console.error('Error processing message:', e);
        }
    };
    
    // Handle window resize
    window.addEventListener('resize', () => {
        if (plotInitialized) {
            Plotly.Plots.resize(plotDiv);
        }
    });
    </script>
</body>
</html>"""

@app.route("/stream")
def stream():
    """Server-Sent Events endpoint."""
    @stream_with_context
    def event_stream():
        while True:
            try:
                # Get with timeout to allow checking for client disconnect
                line = event_queue.get(timeout=1.0)
                yield f"data: {line}\n\n"
            except queue.Empty:
                # Send keepalive comment to detect disconnected clients
                yield ": keepalive\n\n"
            except GeneratorExit:
                # Client disconnected
                break
    
    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"  # Disable Nginx buffering
        }
    )

@app.route("/refresh", methods=["POST"])
def refresh():
    """Endpoint to trigger a refresh of the latest file."""
    try:
        # Signal the file watcher to refresh
        refresh_queue.put("refresh")
        return jsonify({"status": "success", "message": "Refresh initiated"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def run_dashboard():
    """Start the metrics dashboard."""
    # Create metrics directory if it doesn't exist
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Start file watcher thread
    watcher_thread = threading.Thread(target=_file_watcher, daemon=True)
    watcher_thread.start()
    
    print(f"Starting dashboard on http://localhost:{PORT}")
    print(f"Link utilization view: http://localhost:{PORT}/links")
    print(f"Watching for metrics files in: {METRICS_DIR}")
    
    # Run Flask app
    app.run(host="0.0.0.0", port=PORT, threaded=True, debug=False)

if __name__ == "__main__":
    run_dashboard()