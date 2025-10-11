# web_log_viewer.py
"""
Real-time Web Log Viewer for Ryu Controller Modules
Runs on port 8181 to display live logs from all modules with filtering and tabs
"""

import logging
import json
import time
from datetime import datetime
from collections import deque
from threading import Lock
import socketserver
import http.server
from urllib.parse import urlparse, parse_qs
import threading
import os
import signal

from ryu.base import app_manager
from ryu.lib import hub
import setting

class LogHandler(logging.Handler):
    """
    Custom logging handler that captures logs from all Ryu modules
    and stores them for web display
    """
    def __init__(self, log_viewer):
        super().__init__()
        self.log_viewer = log_viewer
        # Rate limiting: track last log times per module
        self.last_log_times = {}
        self.rate_limit_interval = 0.1  # Minimum 100ms between logs per module

    def emit(self, record):
        """Enhanced emit with rate limiting to prevent log flooding"""
        try:
            module_name = self._extract_module_name(record.name)
            current_time = time.time()
            
            # Check rate limiting per module
            if module_name in self.last_log_times:
                time_since_last = current_time - self.last_log_times[module_name]
                if time_since_last < self.rate_limit_interval:
                    # Skip this log due to rate limiting
                    return
            
            self.last_log_times[module_name] = current_time
            
            # Format the log entry (existing code...)
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                'module': module_name,
                'level': record.levelname,
                'message': record.getMessage(),
                'filename': getattr(record, 'filename', ''),
                'lineno': getattr(record, 'lineno', ''),
                'full_name': record.name
            }
            
            # Add to log viewer
            self.log_viewer.add_log(log_entry)
        except Exception as e:
            # Log the error but don't let it break the application
            print(f"LogHandler error: {e}")
    def emit1(self, record):
        """Called when a log message is emitted"""
        try:
            # Format the log entry
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                'module': self._extract_module_name(record.name),
                'level': record.levelname,
                'message': record.getMessage(),
                'filename': getattr(record, 'filename', ''),
                'lineno': getattr(record, 'lineno', ''),
                'full_name': record.name
            }
            
            # Add to log viewer
            self.log_viewer.add_log(log_entry)
        except Exception:
            pass  # Don't let logging errors break the application
    
    def _extract_module_name(self, logger_name):
        """Extract meaningful module name from logger name"""
        # Handle common Ryu logger patterns
        if 'ryu.app.' in logger_name:
            parts = logger_name.split('.')
            if len(parts) >= 3:
                return parts[2]  # Get the app name after 'ryu.app.'
        
        # Handle direct module names and class names
        module_mapping = {
            'FlowRateModule': 'FlowRateModule',
            'NetworkMonitor': 'NetworkMonitor', 
            'GraphStatsMonitor': 'GraphStatsMonitor',
            'Agent2Interface': 'Agent2Interface',
            'Forwarding': 'Forwarding',
            'NetworkMonitoringAPI': 'REST_API',
            'TopologyManagerV2': 'TopologyManager',
            'TopologyManager': 'TopologyManager',
            'detector': 'DelayDetector',
            'DelayDetector': 'DelayDetector',
            'rest_api_v2': 'REST_API',
            'topology_manager': 'TopologyManager',
            'network_monitor': 'NetworkMonitor',
            'forwarding': 'Forwarding',
            'graph_stats_monitor': 'GraphStatsMonitor',
            'agent2_interface': 'Agent2Interface',
            'flow_rate_monitor': 'FlowRateModule',
            'web_log_viewer': 'WebLogViewer'
        }
        
        # Try exact matches first
        for key, value in module_mapping.items():
            if key.lower() in logger_name.lower():
                return value
        
        # Check for partial matches in logger name
        for key, value in module_mapping.items():
            if key in logger_name:
                return value
                
        # Fallback to last part of logger name
        fallback = logger_name.split('.')[-1] if '.' in logger_name else logger_name
        
        # Clean up common suffixes
        if fallback.endswith('Module'):
            return fallback
        elif fallback.endswith('Monitor'):
            return fallback  
        elif fallback.endswith('Manager'):
            return fallback
        else:
            return fallback.capitalize() if fallback else 'Unknown'
        
    

class LogViewerHTTPHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for serving the web interface and log data"""
    
    def __init__(self, log_viewer, *args, **kwargs):
        self.log_viewer = log_viewer
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Serve requests with static file support"""
        try:
            parsed_path = urlparse(self.path)
            
            if parsed_path.path == '/':
                self._serve_main_page()
            elif parsed_path.path == '/debug':
                self._serve_debug_page()
            elif parsed_path.path == '/logs':
                self._serve_logs()
            elif parsed_path.path == '/events':
                self._serve_events()
            elif parsed_path.path == '/modules':
                self._serve_modules()
            # ADD THESE STATIC FILE ROUTES:
            elif parsed_path.path.startswith('/static/'):
                self._serve_static_file(parsed_path.path)
            else:
                self._serve_404()
        except Exception as e:
            self.log_viewer.logger.error(f"Error handling request: {e}")
            self._serve_error_page(f"Internal server error: {e}")
    def _serve_static_file(self, path):
        """Serve static files (CSS/JS)"""
        try:
            import os
            
            # Extract filename and determine content type
            filename = os.path.basename(path)
            if filename.endswith('.css'):
                content_type = 'text/css'
                target_file = 'main.css'
            elif filename.endswith('.js'):
                content_type = 'application/javascript'
                target_file = 'main.js'
            else:
                self._serve_404()
                return
            
            # Look for file in same directory as script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(script_dir, target_file)
            
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self.log_viewer.logger.info(f"Serving static file: {file_path}")
                self.safe_write_response(content, content_type)
            else:
                self.log_viewer.logger.error(f"Static file not found: {file_path}")
                self._serve_404()
                
        except Exception as e:
            self.log_viewer.logger.error(f"Error serving static file: {e}")
            self._serve_error_response(500, f"Error loading file: {e}")

    def _serve_css_file(self, path):
        """
        Serve CSS files from the same directory as the script
        Handles /static/css/main.css requests
        """
        try:
            import os
            
            # Extract filename from path (e.g., main.css from /static/css/main.css)
            filename = os.path.basename(path)
            
            # Look for CSS file in multiple locations
            script_dir = os.path.dirname(os.path.abspath(__file__))
            css_paths = [
                os.path.join(script_dir, 'web', 'static', 'css', filename),
                os.path.join(script_dir, 'static', 'css', filename),
                os.path.join(script_dir, filename),  # Same directory as script
                os.path.join(script_dir, 'main.css')  # Explicit main.css
            ]
            
            css_content = None
            for css_path in css_paths:
                if os.path.exists(css_path):
                    with open(css_path, 'r', encoding='utf-8') as f:
                        css_content = f.read()
                    self.log_viewer.logger.debug(f"Serving CSS from: {css_path}")
                    break
            
            if css_content:
                self.safe_write_response(css_content, 'text/css')
            else:
                self.log_viewer.logger.error(f"CSS file not found: {filename}")
                self.log_viewer.logger.debug(f"Searched paths: {css_paths}")
                self._serve_404()
                
        except Exception as e:
            self.log_viewer.logger.error(f"Error serving CSS file: {e}")
            self._serve_error_response(500, f"Error loading CSS: {e}")

    def _serve_js_file(self, path):
        """
        Serve JavaScript files from the same directory as the script
        Handles /static/js/main.js requests
        """
        try:
            import os
            
            # Extract filename from path (e.g., main.js from /static/js/main.js)
            filename = os.path.basename(path)
            
            # Look for JS file in multiple locations
            script_dir = os.path.dirname(os.path.abspath(__file__))
            js_paths = [
                os.path.join(script_dir, 'web', 'static', 'js', filename),
                os.path.join(script_dir, 'static', 'js', filename),
                os.path.join(script_dir, filename),  # Same directory as script
                os.path.join(script_dir, 'main.js')  # Explicit main.js
            ]
            
            js_content = None
            for js_path in js_paths:
                if os.path.exists(js_path):
                    with open(js_path, 'r', encoding='utf-8') as f:
                        js_content = f.read()
                    self.log_viewer.logger.debug(f"Serving JS from: {js_path}")
                    break
            
            if js_content:
                self.safe_write_response(js_content, 'application/javascript')
            else:
                self.log_viewer.logger.error(f"JS file not found: {filename}")
                self.log_viewer.logger.debug(f"Searched paths: {js_paths}")
                self._serve_404()
                
        except Exception as e:
            self.log_viewer.logger.error(f"Error serving JS file: {e}")
            self._serve_error_response(500, f"Error loading JS: {e}")
    def _serve_main_page(self):
        """
        Serve the main HTML page from template file
        Now with working infrastructure
        """
        try:
            # Get the path to the template file
            template_path = self._get_template_path()
            
            # Read the HTML template
            html_content = self._load_template(template_path)
            
            # Send response using our working safe_write_response method
            if not self.safe_write_response(html_content, 'text/html'):
                self.log_viewer.logger.error("Failed to write template response")
                
        except FileNotFoundError as e:
            self.log_viewer.logger.error(f"Template file not found: {e}")
            self._serve_fallback_page()
        except Exception as e:
            self.log_viewer.logger.error(f"Error serving main page: {e}")
            self._serve_error_page(f"Error loading page: {e}")

    def _get_template_path(self):
        """
        Get the path to the HTML template file
        Looks in multiple locations for flexibility
        """
        import os
        
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Possible template locations (in order of preference)
        template_paths = [
            os.path.join(script_dir, 'web', 'templates', 'web_log_viewer.html'),
            os.path.join(script_dir, 'templates', 'web_log_viewer.html'),
            os.path.join(script_dir, 'web_log_viewer.html'),
            os.path.join(os.path.dirname(script_dir), 'web', 'templates', 'web_log_viewer.html')
        ]
        
        # Find the first existing template file
        for path in template_paths:
            if os.path.exists(path):
                self.log_viewer.logger.debug(f"Using template: {path}")
                return path
        
        # If no template found, raise error with helpful message
        raise FileNotFoundError(f"Template not found in any of these locations: {template_paths}")

    def _load_template(self, template_path):
        """
        Load and optionally process the HTML template
        Can be extended to support template variables in the future
        """
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Future enhancement: replace template variables here
            # html_content = html_content.replace('{{PORT}}', str(self.log_viewer.web_port))
            
            return html_content
            
        except Exception as e:
            raise Exception(f"Failed to load template {template_path}: {e}")

    def _serve_fallback_page(self):
        """
        Serve a simple fallback page when template is not found
        Ensures the viewer still works even without the template file
        """
        fallback_html = """<!DOCTYPE html>
    <html>
    <head>
        <title>Ryu Controller - Log Viewer</title>
        <style>
            body { font-family: Arial, sans-serif; background: #1a1a1a; color: white; padding: 20px; }
            .error { color: #ff4444; }
            .info { color: #4CAF50; }
        </style>
    </head>
    <body>
        <h1>🚀 Ryu Controller - Log Viewer</h1>
        <div class="error">⚠️ Template file not found</div>
        <div class="info">Basic functionality available:</div>
        <ul>
            <li><a href="/logs" style="color: #4CAF50;">View Logs (JSON)</a></li>
            <li><a href="/modules" style="color: #4CAF50;">View Modules (JSON)</a></li>
            <li><a href="/events" style="color: #4CAF50;">Event Stream</a></li>
        </ul>
        <p>Please create the template file at: <code>web/templates/web_log_viewer.html</code></p>
    </body>
    </html>"""
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(fallback_html.encode('utf-8'))

    def _serve_error_page(self, error_message):
        """
        Serve an error page when something goes wrong
        """
        error_html = f"""<!DOCTYPE html>
    <html>
    <head>
        <title>Error - Ryu Log Viewer</title>
        <style>
            body {{ font-family: Arial, sans-serif; background: #1a1a1a; color: white; padding: 20px; }}
            .error {{ color: #ff4444; }}
        </style>
    </head>
    <body>
        <h1>Error</h1>
        <div class="error">{error_message}</div>
        <p><a href="/" style="color: #4CAF50;">Try Again</a></p>
    </body>
    </html>"""
        
        self.send_response(500)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(error_html.encode('utf-8'))
    
    def _serve_logs(self):
        """Serve current logs as JSON with error handling"""
        try:
            logs_data = {
                'logs': list(self.log_viewer.get_logs()),
                'total': len(self.log_viewer.logs)
            }
            
            json_content = json.dumps(logs_data)
            
            # Use the corrected safe_write_response method
            if not self.safe_write_response(json_content, 'application/json'):
                return  # Client disconnected
                
        except Exception as e:
            self.log_viewer.logger.error(f"Error serving logs: {e}")
            self._serve_error_response(500, "Internal server error")
    
    def _serve_events(self):
        """
        Serve Server-Sent Events with immediate log delivery
        """
        try:
            # Set SSE headers
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Connection', 'keep-alive')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Add client to active list
            self.log_viewer.add_client(self)
            self.log_viewer.logger.info("New SSE client connected")
            
            # Send connection confirmation
            self.wfile.write(b'data: {"type":"connected"}\n\n')
            self.wfile.flush()
            
            # Send all existing logs immediately
            existing_logs = self.log_viewer.get_logs()
            self.log_viewer.logger.info(f"Sending {len(existing_logs)} existing logs to new client")
            
            for log_entry in existing_logs:
                try:
                    message = f"data: {json.dumps(log_entry)}\n\n"
                    self.wfile.write(message.encode('utf-8'))
                    self.wfile.flush()
                except Exception as e:
                    self.log_viewer.logger.error(f"Error sending existing log: {e}")
                    break
            
            # Keep connection alive and wait for new logs
            heartbeat_counter = 0
            while True:
                try:
                    time.sleep(1)
                    heartbeat_counter += 1
                    
                    # Send heartbeat every 30 seconds
                    if heartbeat_counter >= 30:
                        self.wfile.write(b': heartbeat\n\n')
                        self.wfile.flush()
                        heartbeat_counter = 0
                        
                except (BrokenPipeError, ConnectionResetError, OSError):
                    self.log_viewer.logger.info("SSE client disconnected")
                    break
                except Exception as e:
                    self.log_viewer.logger.error(f"SSE error: {e}")
                    break
                    
        except Exception as e:
            self.log_viewer.logger.error(f"SSE setup error: {e}")
        finally:
            self.log_viewer.remove_client(self)
            self.log_viewer.logger.info("SSE client removed")
    
    def _serve_modules(self):
        """Serve available modules with real-time counts"""
        try:
            # Get dynamic module counts
            module_counts = self.log_viewer.get_active_modules()
            
            # Add metadata about filtering
            modules_data = {
                'modules': module_counts,
                'total_modules': len(module_counts),
                'active_modules': len([m for m, count in module_counts.items() if count > 0]),
                'timestamp': datetime.now().isoformat()
            }
            
            json_content = json.dumps(modules_data)
            
            if not self.safe_write_response(json_content, 'application/json'):
                return  # Client disconnected
                
        except Exception as e:
            self.log_viewer.logger.error(f"Error serving modules: {e}")
            self._serve_error_response(500, "Internal server error")
    
    def _serve_404(self):
        """Serve 404 error"""
        self.send_response(404)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'Not Found')
    
    def log_message(self, format, *args):
        """
        Override to suppress normal HTTP logs but log errors
        Reduces noise in console output
        """
        # Only log errors and important events
        if any(word in format % args for word in ['error', 'Error', 'ERROR']):
            self.log_viewer.logger.warning(f"HTTP: {format % args}")
    def safe_write_response(self, content, content_type='text/html', status_code=200):
        """
        Safely write response to client with proper error handling
        Prevents BrokenPipeError when client disconnects early
        """
        try:
            self.send_response(status_code)
            
            # Standard headers
            self.send_header('Content-type', content_type)
            
            # Convert content to bytes if it's a string
            if isinstance(content, str):
                content_bytes = content.encode('utf-8')
            else:
                content_bytes = content
                
            self.send_header('Content-Length', str(len(content_bytes)))
            
            # Add caching headers for better performance
            if content_type == 'text/html':
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Expires', '0')
            elif content_type == 'application/json':
                self.send_header('Cache-Control', 'no-cache')
                
            # Security headers
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('X-Frame-Options', 'DENY')
            
            # CORS headers for API endpoints
            if self.path.startswith('/logs') or self.path.startswith('/events'):
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
                
            self.end_headers()
            
            # Write content in chunks to handle large responses
            chunk_size = 8192  # 8KB chunks
            for i in range(0, len(content_bytes), chunk_size):
                chunk = content_bytes[i:i + chunk_size]
                self.wfile.write(chunk)
                self.wfile.flush()  # Ensure data is sent immediately
                
            return True
            
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            self.log_viewer.logger.debug(f"Client disconnected: {e}")
            return False
        except Exception as e:
            self.log_viewer.logger.error(f"Error writing response: {e}")
            return False
        
    def _serve_error_response(self, status_code, message):
        """
        Serve error response safely
        Used when main content serving fails
        """
        try:
            error_content = f"""
            <!DOCTYPE html>
            <html>
            <head><title>Error {status_code}</title></head>
            <body>
                <h1>Error {status_code}</h1>
                <p>{message}</p>
            </body>
            </html>
            """
            self.safe_write_response(error_content, 'text/html', status_code)
        except Exception:
            # If even error response fails, just close connection
            pass

    def _serve_debug_page(self):
        """
        Simple debug page to test if server is working
        Minimal HTML without complex JavaScript
        """
        simple_html = """<!DOCTYPE html>
    <html>
    <head>
        <title>Debug Test</title>
    </head>
    <body>
        <h1>Debug Test Page</h1>
        <p>If you see this, the server is working!</p>
        <p>Current time: """ + str(datetime.now()) + """</p>
    </body>
    </html>"""
        
        try:
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Content-Length', str(len(simple_html.encode())))
            self.end_headers()
            self.wfile.write(simple_html.encode())
            return True
        except Exception as e:
            print(f"Debug page error: {e}")
            return False

class WebLogViewer(app_manager.RyuApp):
    """
    Web-based real-time log viewer for Ryu controller
    """
    
    def __init__(self, *args, **kwargs):
        super(WebLogViewer, self).__init__(*args, **kwargs)
        self.name = 'WebLogViewer'
        
        # Configuration
        self.web_port = getattr(setting, 'LOG_VIEWER_PORT', 8181)
        self.max_logs = getattr(setting, 'MAX_LOG_ENTRIES', 1000)
        
        # Log storage - thread-safe
        self.logs = deque(maxlen=self.max_logs)
        self.logs_lock = Lock()
        
        # Track connected clients for Server-Sent Events
        self.clients = []
        self.clients_lock = Lock()
        
        # Set up custom log handler to capture all Ryu logs
        self.setup_log_capture()
        
        # Start web server in separate thread
        self.web_thread = hub.spawn(self.start_web_server)
        
        self.logger.info(f"Web Log Viewer started on port {self.web_port}")
        print(f"🌐 Access logs at: http://localhost:{self.web_port}")
    
    def setup_log_capture(self):
        """Set up logging handler to capture all Ryu module logs"""
        # Create our custom handler
        log_handler = LogHandler(self)
        log_handler.setLevel(logging.DEBUG)
        
        # Add handler to root logger to catch all logs
        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)
        
        # Also add to specific Ryu loggers to ensure we catch everything
        ryu_loggers = [
            'ryu.app.FlowRateModule',
            'ryu.app.NetworkMonitor',
            'ryu.app.GraphStatsMonitor', 
            'ryu.app.Agent2Interface',
            'ryu.app.Forwarding',
            'ryu.app.rest_api_v2',
            'ryu.app.topology_manager',
            'ryu.app.detector'
        ]
        
        for logger_name in ryu_loggers:
            logger = logging.getLogger(logger_name)
            logger.addHandler(log_handler)
            logger.setLevel(logging.DEBUG)
    
    def add_log(self, log_entry):
        """Add a log entry and notify clients"""
        with self.logs_lock:
            self.logs.append(log_entry)
        
        # Notify all connected clients
        self.notify_clients(log_entry)
    
    def get_logs(self, count=None):
        """Get recent logs"""
        with self.logs_lock:
            if count:
                return list(self.logs)[-count:]
            return list(self.logs)
    
    def add_client(self, client):
        """Add a client for Server-Sent Events"""
        with self.clients_lock:
            self.clients.append(client)
    
    def remove_client(self, client):
        """Remove a client"""
        with self.clients_lock:
            if client in self.clients:
                self.clients.remove(client)
    
    def notify_clients(self, log_entry):
        """Send log entry to all connected clients"""
        message = f"data: {json.dumps(log_entry)}\n\n"
        
        with self.clients_lock:
            clients_to_remove = []
            for client in self.clients:
                try:
                    client.wfile.write(message.encode())
                    client.wfile.flush()
                except Exception:
                    # Client disconnected
                    clients_to_remove.append(client)
            
            # Remove disconnected clients
            for client in clients_to_remove:
                self.clients.remove(client)
    
    def start_web_server(self):
        """Start the HTTP server for the web interface"""
        class CustomHTTPServer(socketserver.TCPServer):
            allow_reuse_address = True
        
        def handler_factory(log_viewer):
            def handler(*args, **kwargs):
                LogViewerHTTPHandler(log_viewer, *args, **kwargs)
            return handler
        
        try:
            with CustomHTTPServer(("", self.web_port), handler_factory(self)) as httpd:
                self.logger.info(f"Web server listening on port {self.web_port}")
                httpd.serve_forever()
        except Exception as e:
            self.logger.error(f"Failed to start web server: {e}")
            print(f"❌ Failed to start web server on port {self.web_port}: {e}")

    def manage_client_connections(self):
        """
        Monitor and cleanup stale client connections
        Prevents memory leaks from disconnected clients
        """
        with self.clients_lock:
            active_clients = []
            for client in self.clients:
                try:
                    # Send a heartbeat to test connection
                    client.wfile.write(b': heartbeat\n\n')
                    client.wfile.flush()
                    active_clients.append(client)
                except Exception:
                    # Client is disconnected, don't add to active list
                    self.logger.debug("Removed stale client connection")
            
            # Update clients list with only active connections
            removed_count = len(self.clients) - len(active_clients)
            self.clients.clear()
            self.clients.extend(active_clients)
            
            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} stale client connections")

    # Add periodic connection management to __init__
    def _start_connection_manager(self):
        """Start background thread for connection management"""
        def connection_manager():
            while not getattr(self, '_shutdown', False):
                try:
                    self.manage_client_connections()
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    self.logger.error(f"Connection management error: {e}")
                    time.sleep(60)  # Wait 1 minute before retry
        
        self.connection_thread = hub.spawn(connection_manager)

    def cleanup_old_logs(self):
        """
        Periodic cleanup of old logs to prevent memory buildup
        Should be called periodically (e.g., every hour)
        """
        with self.logs_lock:
            # Keep only logs from last 24 hours
            cutoff_time = time.time() - (24 * 60 * 60)  # 24 hours ago
            
            # Convert to list for iteration and filtering
            current_logs = list(self.logs)
            valid_logs = []
            
            for log_entry in current_logs:
                try:
                    # Parse timestamp and check if it's within retention period
                    log_time = datetime.strptime(log_entry['timestamp'], "%Y-%m-%d %H:%M:%S.%f")
                    if log_time.timestamp() > cutoff_time:
                        valid_logs.append(log_entry)
                except (ValueError, KeyError):
                    # Keep logs with invalid timestamps to avoid data loss
                    valid_logs.append(log_entry)
            
            # Clear and repopulate with valid logs
            self.logs.clear()
            self.logs.extend(valid_logs)
            
            self.logger.info(f"Log cleanup: {len(current_logs) - len(valid_logs)} old entries removed")

    # Add this to __init__ method to start periodic cleanup
    def _start_cleanup_thread(self):
        """Start background thread for periodic log cleanup"""
        def cleanup_worker():
            while not getattr(self, '_shutdown', False):
                try:
                    self.cleanup_old_logs()
                    time.sleep(3600)  # Run every hour
                except Exception as e:
                    self.logger.error(f"Log cleanup error: {e}")
                    time.sleep(300)  # Wait 5 minutes before retry
        
        self.cleanup_thread = hub.spawn(cleanup_worker)

    def shutdown(self):
        """
        Graceful shutdown of web log viewer
        Closes all connections and stops background threads
        """
        self.logger.info("Shutting down Web Log Viewer...")
        
        # Set shutdown flag
        self._shutdown = True
        
        # Close all client connections
        with self.clients_lock:
            for client in self.clients:
                try:
                    client.wfile.close()
                except Exception:
                    pass
            self.clients.clear()
        
        # Stop background threads
        if hasattr(self, 'cleanup_thread'):
            self.cleanup_thread.kill()
        if hasattr(self, 'connection_thread'):
            self.connection_thread.kill()
        if hasattr(self, 'web_thread'):
            self.web_thread.kill()
        
        self.logger.info("Web Log Viewer shutdown complete")

    def get_active_modules(self):
        """
        Get list of active modules with their log counts
        Updates dynamically based on actual logging activity
        """
        module_counts = {}
        
        with self.logs_lock:
            for log_entry in self.logs:
                module_name = log_entry['module']
                module_counts[module_name] = module_counts.get(module_name, 0) + 1
        
        # Add any known modules even if they haven't logged yet
        known_modules = [
            'FlowRateModule', 'NetworkMonitor', 'GraphStatsMonitor',
            'Agent2Interface', 'Forwarding', 'REST_API', 
            'TopologyManager', 'DelayDetector', 'WebLogViewer'
        ]
        
        for module in known_modules:
            if module not in module_counts:
                module_counts[module] = 0
        
        return module_counts
    def get_active_modules(self):
        """Get module counts for the /modules endpoint"""
        module_counts = {}
        
        with self.logs_lock:
            for log_entry in self.logs:
                module_name = log_entry.get('module', 'Unknown')
                module_counts[module_name] = module_counts.get(module_name, 0) + 1
        
        return module_counts
    # Add signal handler to __init__ for graceful shutdown
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)