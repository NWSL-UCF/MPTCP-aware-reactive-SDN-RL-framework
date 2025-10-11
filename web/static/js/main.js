/**
 * Ryu Controller Log Viewer - Main JavaScript
 * Real-time log viewer with filtering and module management
 */

// Global Application State
const LogViewer = {
    // Data
    allLogs: [],
    filteredLogs: [],
    modules: {},
    
    // UI State
    currentModule: 'all',
    autoScroll: true,
    
    // Connection
    eventSource: null,
    moduleUpdateInterval: null,
    
    // Configuration
    config: {
        maxDisplayLogs: 500,
        moduleUpdateInterval: 5000,
        reconnectDelay: 5000
    }
};

/**
 * Initialize the application when DOM is loaded
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Initializing Ryu Log Viewer...');
    
    LogViewer.initializeApplication();
});

/**
 * Main application initialization
 */
LogViewer.initializeApplication = function() {
    try {
        this.loadModules();
        this.setupEventSource();
        this.setupFilters();
        this.initializeModuleUpdates();
        this.updateStats();
        this.setupKeyboardShortcuts();
        
        console.log('✅ Log Viewer initialized successfully');
    } catch (error) {
        console.error('❌ Failed to initialize Log Viewer:', error);
        this.showError('Failed to initialize application');
    }
};

/**
 * Module Management Functions
 */

LogViewer.loadModules = function() {
    fetch('/modules')
        .then(response => {
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return response.json();
        })
        .then(data => {
            this.modules = data.modules;
            this.updateModuleFilter();
            this.updateTabs();
            this.updateModuleStats(data);
        })
        .catch(error => {
            console.error('Error loading modules:', error);
            this.showError('Failed to load modules');
        });
};

LogViewer.initializeModuleUpdates = function() {
    // Update modules periodically
    this.moduleUpdateInterval = setInterval(() => {
        this.loadModules();
    }, this.config.moduleUpdateInterval);
};

LogViewer.updateModuleFilter = function() {
    const moduleFilter = document.getElementById('moduleFilter');
    if (!moduleFilter) return;
    
    const currentValue = moduleFilter.value;
    
    // Calculate total logs for "All Modules"
    const totalLogs = Object.values(this.modules).reduce((sum, count) => sum + count, 0);
    
    moduleFilter.innerHTML = `<option value="all">All Modules (${totalLogs})</option>`;
    
    // Sort modules by count (descending) and then alphabetically
    const sortedModules = Object.entries(this.modules)
        .sort((a, b) => {
            if (b[1] !== a[1]) return b[1] - a[1]; // Sort by count first
            return a[0].localeCompare(b[0]); // Then alphabetically
        });
    
    sortedModules.forEach(([module, count]) => {
        const option = document.createElement('option');
        option.value = module;
        option.textContent = `${module} (${count})`;
        
        // Add visual styling for active modules
        if (count > 0) {
            option.style.fontWeight = 'bold';
            option.style.color = '#4CAF50';
        } else {
            option.style.color = '#666';
        }
        
        moduleFilter.appendChild(option);
    });
    
    // Restore previous selection if it still exists
    if (currentValue && [...moduleFilter.options].some(opt => opt.value === currentValue)) {
        moduleFilter.value = currentValue;
    }
};

LogViewer.updateModuleStats = function(data) {
    // Update any module statistics displays
    if (data.total_modules !== undefined) {
        console.log(`📊 Modules: ${data.total_modules} total, ${data.active_modules} active`);
    }
};

/**
 * Real-time Connection Management
 */

LogViewer.setupEventSource = function() {
    const statusIndicator = document.getElementById('statusIndicator');
    
    // Load existing logs first
    this.loadExistingLogs();
    
    if (this.eventSource) {
        this.eventSource.close();
    }
    
    console.log('🔌 Connecting to event stream...');
    this.eventSource = new EventSource('/events');
    
    this.eventSource.onopen = () => {
        if (statusIndicator) statusIndicator.classList.add('connected');
        console.log('✅ Connected to log stream');
        this.clearError();
    };
    
    this.eventSource.onmessage = (event) => {
        try {
            const logEntry = JSON.parse(event.data);
            
            // Skip connection confirmation messages
            if (logEntry.type === 'connected') {
                console.log('🔗 EventSource connection confirmed');
                return;
            }
            
            // Add new log entry
            this.addLogEntry(logEntry);
        } catch (error) {
            console.error('Error parsing log entry:', error);
        }
    };
    
    this.eventSource.onerror = (event) => {
        if (statusIndicator) statusIndicator.classList.remove('connected');
        console.log('🔄 Connection lost, retrying...');
        
        setTimeout(() => {
            this.setupEventSource();
        }, this.config.reconnectDelay);
    };
};
LogViewer.setupEventSource = function() {
    // ... existing code (don't change this yet) ...
};

// ADD THIS NEW FUNCTION HERE:
LogViewer.loadExistingLogs = function() {
    console.log('📥 Loading existing logs...');
    fetch('/logs')
        .then(response => response.json())
        .then(data => {
            console.log(`📦 Loaded ${data.logs.length} existing logs`);
            
            // Clear the "Connecting..." message
            const logContent = document.getElementById('logContent');
            if (logContent) {
                logContent.innerHTML = '';
            }
            
            // Add each log
            data.logs.forEach(logEntry => {
                this.addLogEntry(logEntry);
            });
            
            console.log('✅ Existing logs loaded successfully');
        })
        .catch(error => {
            console.error('❌ Error loading existing logs:', error);
        });
};

/**
 * Log Entry Management
 */

LogViewer.addLogEntry = function(logEntry) {
    // Add to all logs
    this.allLogs.push(logEntry);
    
    // Update module counts
    if (this.modules[logEntry.module] !== undefined) {
        this.modules[logEntry.module]++;
    } else {
        this.modules[logEntry.module] = 1;
    }
    
    // Update displays
    this.filterAndDisplayLogs();
    this.updateTabs();
    this.updateStats();
};

LogViewer.filterAndDisplayLogs = function() {
    const levelFilter = document.getElementById('levelFilter');
    const searchFilter = document.getElementById('searchFilter');
    
    if (!levelFilter || !searchFilter) {
        console.log('❌ Missing filter elements');
        return;
    }
    
    const levelValue = levelFilter.value;
    const searchValue = searchFilter.value.toLowerCase();
    
    // Handle both module filter systems
    let moduleValue = 'all';
    const moduleFilter = document.getElementById('moduleFilter');
    
    if (moduleFilter) {
        moduleValue = moduleFilter.value;
    } else if (window.selectedModules) {
        // Use the custom dropdown system
        moduleValue = selectedModules.has('all') ? 'all' : Array.from(selectedModules)[0];
    }
    
    console.log(`🔍 Filtering: Module=${moduleValue}, Level=${levelValue}, Search="${searchValue}"`);
    
    this.filteredLogs = this.allLogs.filter(log => {
        let moduleMatch;
        
        // Handle both module filtering systems
        if (window.selectedModules && !selectedModules.has('all')) {
            moduleMatch = selectedModules.has(log.module);
        } else {
            moduleMatch = moduleValue === 'all' || log.module === moduleValue;
        }
        
        const levelMatch = !levelValue || log.level === levelValue;
        const searchMatch = !searchValue || 
            log.message.toLowerCase().includes(searchValue) ||
            log.module.toLowerCase().includes(searchValue);
        
        return moduleMatch && levelMatch && searchMatch;
    });
    
    console.log(`📊 Filtered ${this.filteredLogs.length} logs from ${this.allLogs.length} total`);
    this.displayLogs();
};

LogViewer.displayLogs = function() {
    const logContent = document.getElementById('logContent');
    if (!logContent) return;
    
    if (this.filteredLogs.length === 0) {
        logContent.innerHTML = '<div class="no-logs">No logs match the current filters</div>';
        return;
    }
    
    // Limit displayed logs for performance
    const logsToDisplay = this.filteredLogs.slice(-this.config.maxDisplayLogs);
    
    logContent.innerHTML = logsToDisplay.map(log => `
        <div class="log-entry" data-level="${log.level}" data-module="${log.module}">
            <div class="log-timestamp">${log.timestamp}</div>
            <div class="log-module">${log.module}</div>
            <div class="log-level ${log.level}">${log.level}</div>
            <div class="log-message">${this.escapeHtml(log.message)}</div>
        </div>
    `).join('');
    
    if (this.autoScroll) {
        this.scrollToBottom();
    } else {
        this.updateScrollIndicator();
    }
    
    this.updateStats();
};

/**
 * Tab Management
 */

LogViewer.updateTabs = function() {
    const tabContainer = document.getElementById('tabContainer');
    if (!tabContainer) return;
    
    const tabs = [
        { id: 'all', name: 'All', count: this.allLogs.length, isAll: true }
    ];
    
    // Sort modules by activity level and only show active ones
    const sortedModules = Object.entries(this.modules)
        .filter(([module, count]) => count > 0)
        .sort((a, b) => b[1] - a[1]);
    
    sortedModules.forEach(([module, count]) => {
        tabs.push({
            id: module,
            name: module,
            count: count,
            isAll: false
        });
    });
    
    tabContainer.innerHTML = tabs.map(tab => `
        <button class="tab ${this.currentModule === tab.id ? 'active' : ''}" 
                onclick="LogViewer.selectModule('${tab.id}')"
                title="${tab.isAll ? 'Show all logs' : `Show only ${tab.name} logs`}">
            ${tab.name}
            <span class="tab-count ${tab.count > 0 ? 'has-logs' : ''}">${tab.count}</span>
        </button>
    `).join('');
};

LogViewer.selectModule = function(moduleId) {
    this.currentModule = moduleId;
    
    const moduleFilter = document.getElementById('moduleFilter');
    const currentModuleSpan = document.getElementById('currentModule');
    
    if (moduleFilter) moduleFilter.value = moduleId;
    if (currentModuleSpan) {
        currentModuleSpan.textContent = moduleId === 'all' ? 'All Modules' : moduleId;
    }
    
    this.filterAndDisplayLogs();
    this.updateTabs();
    
    // Clear any active quick filters
    this.clearQuickFilters();
};

/**
 * Filter Functions
 */

LogViewer.setupFilters = function() {
    const moduleFilter = document.getElementById('moduleFilter');
    const levelFilter = document.getElementById('levelFilter');
    const searchFilter = document.getElementById('searchFilter');
    
    if (moduleFilter) {
        moduleFilter.addEventListener('change', () => this.filterAndDisplayLogs());
    }
    if (levelFilter) {
        levelFilter.addEventListener('change', () => this.filterAndDisplayLogs());
    }
    if (searchFilter) {
        searchFilter.addEventListener('input', () => this.filterAndDisplayLogs());
    }
    
    // Setup scroll listener
    const logContent = document.getElementById('logContent');
    if (logContent) {
        logContent.addEventListener('scroll', () => this.updateScrollIndicator());
    }
};

LogViewer.filterByLevel = function(level) {
    const levelFilter = document.getElementById('levelFilter');
    if (levelFilter) {
        levelFilter.value = level;
        this.filterAndDisplayLogs();
    }
    
    // Visual feedback for quick filters
    this.clearQuickFilters();
    const button = event.target;
    if (button) button.classList.add('selected');
};

LogViewer.showActiveModulesOnly = function() {
    // Filter to only show modules with log entries
    const activeModules = Object.entries(this.modules).filter(([module, count]) => count > 0);
    
    if (activeModules.length > 0) {
        // Switch to the first active module
        const firstActiveModule = activeModules[0][0];
        this.selectModule(firstActiveModule);
    }
    
    // Visual feedback
    this.clearQuickFilters();
    const button = event.target;
    if (button) button.classList.add('selected');
};

LogViewer.clearQuickFilters = function() {
    document.querySelectorAll('.quick-filter').forEach(btn => {
        btn.classList.remove('selected');
    });
};

/**
 * UI Control Functions
 */

LogViewer.updateStats = function() {
    const totalLogsSpan = document.getElementById('totalLogs');
    const visibleLogsSpan = document.getElementById('visibleLogs');
    
    if (totalLogsSpan) totalLogsSpan.textContent = this.allLogs.length;
    if (visibleLogsSpan) visibleLogsSpan.textContent = this.filteredLogs.length;
};

LogViewer.clearLogs = function() {
    this.allLogs = [];
    this.filteredLogs = [];
    
    // Reset module counts
    Object.keys(this.modules).forEach(key => {
        this.modules[key] = 0;
    });
    
    this.displayLogs();
    this.updateTabs();
    this.updateStats();
    this.updateModuleFilter();
};

LogViewer.toggleAutoScroll = function() {
    this.autoScroll = !this.autoScroll;
    const btn = document.getElementById('autoScrollBtn');
    
    if (btn) {
        btn.textContent = `Auto-scroll: ${this.autoScroll ? 'ON' : 'OFF'}`;
        btn.style.background = this.autoScroll ? '#4CAF50' : '#666';
    }
    
    if (this.autoScroll) {
        this.scrollToBottom();
    }
};

LogViewer.scrollToBottom = function() {
    const logContent = document.getElementById('logContent');
    if (logContent) {
        logContent.scrollTop = logContent.scrollHeight;
        this.updateScrollIndicator();
    }
};

LogViewer.updateScrollIndicator = function() {
    const logContent = document.getElementById('logContent');
    const scrollIndicator = document.getElementById('scrollIndicator');
    
    if (!logContent || !scrollIndicator) return;
    
    const isAtBottom = logContent.scrollTop + logContent.clientHeight >= logContent.scrollHeight - 5;
    
    if (!isAtBottom && this.filteredLogs.length > 0) {
        scrollIndicator.style.display = 'block';
    } else {
        scrollIndicator.style.display = 'none';
    }
};

/**
 * Keyboard Shortcuts
 */

LogViewer.setupKeyboardShortcuts = function() {
    document.addEventListener('keydown', (event) => {
        // Only handle shortcuts when not typing in input fields
        if (event.target.tagName === 'INPUT' || event.target.tagName === 'SELECT') {
            return;
        }
        
        switch (event.key) {
            case 'c':
                if (event.ctrlKey || event.metaKey) return; // Allow Ctrl+C
                this.clearLogs();
                event.preventDefault();
                break;
            case 's':
                this.toggleAutoScroll();
                event.preventDefault();
                break;
            case 'f':
                const searchInput = document.getElementById('searchFilter');
                if (searchInput) searchInput.focus();
                event.preventDefault();
                break;
            case 'Escape':
                // Clear search and filters
                const searchFilter = document.getElementById('searchFilter');
                const levelFilter = document.getElementById('levelFilter');
                if (searchFilter) searchFilter.value = '';
                if (levelFilter) levelFilter.value = '';
                this.selectModule('all');
                event.preventDefault();
                break;
            case 'End':
                this.scrollToBottom();
                event.preventDefault();
                break;
        }
    });
};

/**
 * Error Handling
 */

LogViewer.showError = function(message) {
    const logContent = document.getElementById('logContent');
    if (logContent) {
        logContent.innerHTML = `<div class="error">⚠️ ${message}</div>`;
    }
    console.error('LogViewer Error:', message);
};

LogViewer.clearError = function() {
    const logContent = document.getElementById('logContent');
    if (logContent && logContent.querySelector('.error')) {
        this.displayLogs();
    }
};

/**
 * Utility Functions
 */

LogViewer.escapeHtml = function(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
};

/**
 * Page Visibility Handling
 */

document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        if (LogViewer.eventSource) LogViewer.eventSource.close();
    } else {
        LogViewer.setupEventSource();
    }
});

/**
 * Global Functions for HTML onclick handlers
 * (Keep these for backward compatibility)
 */

function clearLogs() {
    LogViewer.clearLogs();
}

function toggleAutoScroll() {
    LogViewer.toggleAutoScroll();
}

function scrollToBottom() {
    LogViewer.scrollToBottom();
}

function filterByLevel(level) {
    LogViewer.filterByLevel(level);
}

function showActiveModulesOnly() {
    LogViewer.showActiveModulesOnly();
}

function selectModule(moduleId) {
    LogViewer.selectModule(moduleId);
}

// Export LogViewer for debugging in console
window.LogViewer = LogViewer;