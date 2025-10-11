from flask import Flask, render_template_string
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
import logging
from topology_manager import TopologyManager
from network_monitor import NetworkMonitor

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

topology_manager = TopologyManager()
network_monitor = NetworkMonitor()

@app.route('/')
def home():
    # Update the topology to ensure the latest data
    try:
        topology_manager.update_topology()
    except Exception as e:
        logger.error(f"Error updating topology: {e}")
        return "Error updating topology"

    switches = list(topology_manager.topology.nodes)
    num_switches = len(switches)
    links = list(topology_manager.topology.edges)
    num_links = len(links)

    logger.info(f"Number of switches: {num_switches}")
    logger.info(f"Number of links: {num_links}")
    logger.info(f"Switches: {switches}")
    logger.info(f"Links: {links}")

    G = topology_manager.topology
    img_base64 = ""
    if len(G.nodes) > 0:
        try:
            plt.figure(figsize=(8, 6))
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1500, font_size=10)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            plt.close()
        except Exception as e:
            logger.error(f"Error generating topology image: {e}")
            img_base64 = ""

    # Get all paths from topology manager
    paths = topology_manager.all_paths
    logger.info(f"All paths: {paths}")

    paths_list = []
    if paths:
        try:
            for path_id, path_details in paths.items():
                path_str = ' -> '.join(str(node[0]) for node in path_details)
                details_str = ' | '.join(f"{node[0]} (link: {node[1]})" if node[1] else f"{node[0]}" for node in path_details)
                paths_list.append({'path_id': path_id, 'path': path_str, 'details': details_str})
        except Exception as e:
            logger.error(f"Error processing paths: {e}")

    # HTML Template to display the data
    html_template = """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        <title>Network Topology Dashboard</title>
      </head>
      <body>
        <h1>Network Topology Dashboard</h1>
        <h2>Network Summary</h2>
        <p>Number of Switches: {{ num_switches }}</p>
        <p>Number of Links: {{ num_links }}</p>

        <h2>Network Topology</h2>
        {% if image_data %}
        <img src="data:image/png;base64,{{ image_data }}" alt="Network Topology">
        {% else %}
        <p>No topology data available.</p>
        {% endif %}

        <h2>All Paths</h2>
        {% if paths %}
        <table border="1">
          <thead>
            <tr>
              <th>Path ID</th>
              <th>Path</th>
              <th>Details</th>
            </tr>
          </thead>
          <tbody>
            {% for path in paths %}
            <tr>
              <td>{{ path.path_id }}</td>
              <td>{{ path.path }}</td>
              <td>{{ path.details }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
        {% else %}
        <p>No paths available.</p>
        {% endif %}
      </body>
    </html>
    """

    return render_template_string(html_template, num_switches=num_switches, num_links=num_links, image_data=img_base64, paths=paths_list)

if __name__ == '__main__':
    try:
        logger.info("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.error(f"Failed to start Flask server: {e}")
