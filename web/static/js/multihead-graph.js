class GraphRenderer {
    constructor(container) {
        this.container = container;
        this.canvas = null;
        this.ctx = null;
        this.nodes = [];
        this.edges = [];
        this.selectedNode = null;
        this.isDragging = false;
        this.dragOffset = { x: 0, y: 0 };
        this.camera = { x: 0, y: 0, zoom: 1 };
        this.animationId = null;
        
        this.initCanvas();
        this.bindEvents();
        this.startRenderLoop();
    }
    
    initCanvas() {
        this.canvas = document.createElement('canvas');
        this.canvas.width = this.container.clientWidth;
        this.canvas.height = this.container.clientHeight;
        this.container.appendChild(this.canvas);
        
        this.ctx = this.canvas.getContext('2d');
        
        // Handle resize
        window.addEventListener('resize', () => {
            this.canvas.width = this.container.clientWidth;
            this.canvas.height = this.container.clientHeight;
        });
    }
    
    updateData(nodes, edges) {
        this.nodes = nodes.map(node => ({
            ...node,
            x: node.x || this.canvas.width / 2,
            y: node.y || this.canvas.height / 2,
            vx: 0,
            vy: 0,
            fixed: true
        }));
        
        this.edges = edges;
        console.log(`Updated graph data: ${this.nodes.length} nodes, ${this.edges.length} edges`);
    }
    
    bindEvents() {
        this.canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.onMouseUp(e));
        this.canvas.addEventListener('wheel', (e) => this.onWheel(e));
        this.canvas.addEventListener('click', (e) => this.onClick(e));
    }
    
    getMousePos(e) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: (e.clientX - rect.left - this.camera.x) / this.camera.zoom,
            y: (e.clientY - rect.top - this.camera.y) / this.camera.zoom
        };
    }
    
    getNodeAt(x, y) {
        for (let node of this.nodes) {
            const dx = x - node.x;
            const dy = y - node.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            if (distance < (node.size || 10)) {
                return node;
            }
        }
        return null;
    }
    
    onMouseDown(e) {
        const mousePos = this.getMousePos(e);
        const node = this.getNodeAt(mousePos.x, mousePos.y);
        
        if (node) {
            this.selectedNode = node;
            this.isDragging = true;
            this.dragOffset = {
                x: mousePos.x - node.x,
                y: mousePos.y - node.y
            };
            e.preventDefault();
        } else {
            this.isDragging = true;
            this.dragOffset = { x: mousePos.x, y: mousePos.y };
        }
    }
    
    onMouseMove(e) {
        if (!this.isDragging) return;
        
        const mousePos = this.getMousePos(e);
        
        if (this.selectedNode) {
            this.selectedNode.x = mousePos.x - this.dragOffset.x;
            this.selectedNode.y = mousePos.y - this.dragOffset.y;
        } else {
            this.camera.x += (mousePos.x - this.dragOffset.x) * this.camera.zoom;
            this.camera.y += (mousePos.y - this.dragOffset.y) * this.camera.zoom;
        }
    }
    
    onMouseUp(e) {
        this.isDragging = false;
        this.selectedNode = null;
    }
    
    onWheel(e) {
        e.preventDefault();
        const mousePos = this.getMousePos(e);
        const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
        
        this.camera.x = mousePos.x - (mousePos.x - this.camera.x) * zoomFactor;
        this.camera.y = mousePos.y - (mousePos.y - this.camera.y) * zoomFactor;
        this.camera.zoom *= zoomFactor;
        
        this.camera.zoom = Math.max(0.1, Math.min(3, this.camera.zoom));
    }
    
    onClick(e) {
        const mousePos = this.getMousePos(e);
        const node = this.getNodeAt(mousePos.x, mousePos.y);
        
        if (node) {
            this.showNodeDetails(node);
        }
    }
    
    showNodeDetails(node) {
        const detailsPanel = document.getElementById('nodeDetails');
        const nodeTitle = document.getElementById('nodeTitle');
        const nodeInfo = document.getElementById('nodeInfo');
        
        if (detailsPanel && nodeTitle && nodeInfo) {
            nodeTitle.textContent = `${node.type} - ${node.id}`;
            
            let infoHtml = `
                <p><strong>Head:</strong> ${node.head}</p>
                <p><strong>Type:</strong> ${node.type}</p>
                <p><strong>Size:</strong> ${node.size}</p>
                <p><strong>Layer:</strong> ${node.layer || node.head}</p>
            `;
            
            if (node.utilization !== undefined) {
                infoHtml += `<p><strong>Utilization:</strong> ${node.utilization.toFixed(2)}%</p>`;
            }
            
            if (node.data) {
                for (const [key, value] of Object.entries(node.data)) {
                    if (key !== 'timestamp') {
                        infoHtml += `<p><strong>${key}:</strong> ${value}</p>`;
                    }
                }
            }
            
            nodeInfo.innerHTML = infoHtml;
            detailsPanel.style.display = 'block';
        }
    }
    
    resetView() {
        this.camera = { x: 0, y: 0, zoom: 1 };
        
        this.nodes.forEach(node => {
            if (node.originalX !== undefined && node.originalY !== undefined) {
                node.x = node.originalX;
                node.y = node.originalY;
            }
        });
    }
    
    startRenderLoop() {
        const render = () => {
            this.render();
            this.animationId = requestAnimationFrame(render);
        };
        render();
    }
    
    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.ctx.save();
        this.ctx.translate(this.camera.x, this.camera.y);
        this.ctx.scale(this.camera.zoom, this.camera.zoom);
        
        this.drawLayerGuides();
        this.drawEdges();
        this.drawNodes();
        this.drawLayerLabels();
        
        this.ctx.restore();
    }
    
    drawLayerGuides() {
        const centerX = 400;
        const centerY = 300;
        const layers = {
            1: { radius: 0, color: '#ff6b6b20', label: 'Head 1: Switches' },
            2: { radius: 120, color: '#4ecdc420', label: 'Head 2: Ports' },
            3: { radius: 200, color: '#45b7d120', label: 'Head 3: Flows' },
            4: { radius: 280, color: '#00A14620', label: 'Head 4: New Ports' }
        };
        
        this.ctx.save();
        this.ctx.globalAlpha = 0.3;
        
        for (const [headId, layer] of Object.entries(layers)) {
            if (layer.radius > 0) {
                this.ctx.beginPath();
                this.ctx.arc(centerX, centerY, layer.radius, 0, 2 * Math.PI);
                this.ctx.strokeStyle = layer.color.replace('20', '');
                this.ctx.lineWidth = 1;
                this.ctx.stroke();
                
                this.ctx.fillStyle = layer.color;
                this.ctx.fill();
            }
        }
        
        this.ctx.restore();
    }
    
    drawLayerLabels() {
        const centerX = 400;
        const centerY = 300;
        const layers = {
            1: { radius: 0, label: 'Core' },
            2: { radius: 120, label: 'Layer 1' },
            3: { radius: 200, label: 'Layer 2' },
            4: { radius: 280, label: 'Layer 3' }
        };
        
        this.ctx.save();
        this.ctx.font = '12px Arial';
        this.ctx.fillStyle = '#666';
        this.ctx.textAlign = 'center';
        
        for (const [headId, layer] of Object.entries(layers)) {
            if (layer.radius > 0) {
                this.ctx.fillText(layer.label, centerX, centerY - layer.radius - 10);
            } else {
                this.ctx.fillText(layer.label, centerX, centerY - 30);
            }
        }
        
        this.ctx.restore();
    }
    
    drawEdges() {
        this.ctx.save();
        
        let drawnEdges = 0;
        let head3ToHead4Edges = 0;
        let skippedEdges = 0;
        
        for (const edge of this.edges) {
            // Handle both 'from'/'to' and 'source'/'target' formats
            const fromId = edge.from || edge.source;
            const toId = edge.to || edge.target;
            
            const fromNode = this.nodes.find(n => n.id === fromId);
            const toNode = this.nodes.find(n => n.id === toId);
            
            if (!fromNode || !toNode) {
                console.log(`Cannot draw edge ${fromId} -> ${toId}: missing node(s)`);
                console.log(`  From node found: ${fromNode ? 'yes' : 'no'}`);
                console.log(`  To node found: ${toNode ? 'yes' : 'no'}`);
                console.log(`  Edge type: ${edge.type}`);
                console.log(`  Available nodes: ${this.nodes.length}`);
                skippedEdges++;
                continue;
            }
            
            // Count Head 3 to Head 4 edges
            if (fromNode.head === 3 && toNode.head === 4) {
                head3ToHead4Edges++;
                console.log(`✓ Drawing Head 3->4 edge: ${fromId} -> ${toId}, type: ${edge.type}, weight: ${edge.weight}`);
            }
            
            this.ctx.beginPath();
            this.ctx.moveTo(fromNode.x, fromNode.y);
            this.ctx.lineTo(toNode.x, toNode.y);
            
            // Enhanced visibility for Head 3->4 edges
            if (edge.type === 'flow_to_new_port' || (fromNode.head === 3 && toNode.head === 4)) {
                this.ctx.strokeStyle = '#00ff00';  // Bright green for visibility
                this.ctx.lineWidth = Math.max(4, edge.width || 4);  // Thicker lines
                console.log(`Applied bright green styling to Head 3->4 edge`);
            } else {
                this.ctx.strokeStyle = edge.color || '#999';
                this.ctx.lineWidth = edge.width || 1;
            }
            
            this.ctx.stroke();
            
            // Draw arrow for directed edges
            this.drawArrow(fromNode.x, fromNode.y, toNode.x, toNode.y, this.ctx.strokeStyle);
            drawnEdges++;
        }
        
        console.log(`✓ EDGE SUMMARY: Drew ${drawnEdges} total edges, ${head3ToHead4Edges} Head 3->4 edges, ${skippedEdges} skipped`);
        
        this.ctx.restore();
    }
    
    drawArrow(fromX, fromY, toX, toY, color) {
        const angle = Math.atan2(toY - fromY, toX - fromX);
        const arrowLength = 10;
        const arrowAngle = Math.PI / 6;
        
        const nodeRadius = 10;
        const arrowX = toX - Math.cos(angle) * nodeRadius;
        const arrowY = toY - Math.sin(angle) * nodeRadius;
        
        this.ctx.save();
        this.ctx.strokeStyle = color;
        this.ctx.fillStyle = color;
        this.ctx.lineWidth = 1;
        
        this.ctx.beginPath();
        this.ctx.moveTo(arrowX, arrowY);
        this.ctx.lineTo(
            arrowX - arrowLength * Math.cos(angle - arrowAngle),
            arrowY - arrowLength * Math.sin(angle - arrowAngle)
        );
        this.ctx.moveTo(arrowX, arrowY);
        this.ctx.lineTo(
            arrowX - arrowLength * Math.cos(angle + arrowAngle),
            arrowY - arrowLength * Math.sin(angle + arrowAngle)
        );
        this.ctx.stroke();
        
        this.ctx.restore();
    }
    
    drawNodes() {
        this.ctx.save();
        
        for (const node of this.nodes) {
            this.ctx.beginPath();
            this.ctx.arc(node.x, node.y, node.size || 10, 0, 2 * Math.PI);
            
            this.ctx.fillStyle = node.color || '#666';
            this.ctx.fill();
            
            this.ctx.strokeStyle = this.selectedNode === node ? '#000' : '#333';
            this.ctx.lineWidth = this.selectedNode === node ? 3 : 1;
            this.ctx.stroke();
            
            this.ctx.fillStyle = '#000';
            this.ctx.font = '10px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(node.label || node.id, node.x, node.y + (node.size || 10) + 12);
        }
        
        this.ctx.restore();
    }
    
    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        if (this.canvas && this.canvas.parentNode) {
            this.canvas.parentNode.removeChild(this.canvas);
        }
    }
}

// Debug function for Head 3-4 connections
function debugHead4Data(data) {
    console.log('=== HEAD 3-4 CONNECTION DEBUG (Updated) ===');
    
    const head3Nodes = data.nodes.filter(n => n.head === 3);
    const head4Nodes = data.nodes.filter(n => n.head === 4);
    
    console.log(`Head 3 nodes: ${head3Nodes.length}`);
    console.log('Head 3 node IDs:', head3Nodes.map(n => n.id));
    
    console.log(`Head 4 nodes: ${head4Nodes.length}`);
    console.log('Head 4 node IDs:', head4Nodes.map(n => n.id));
    
    // Find Head 3->4 edges with consistent naming
    const head3ToHead4Edges = data.edges.filter(e => {
        const fromId = e.from || e.source;
        const toId = e.to || e.target;
        
        // Check if source is a Head 3 node and target is a Head 4 node
        const isFromHead3 = head3Nodes.some(n => n.id === fromId);
        const isToHead4 = head4Nodes.some(n => n.id === toId);
        
        return isFromHead3 && isToHead4;
    });
    
    console.log(`Head 3->4 edges found: ${head3ToHead4Edges.length}`);
    head3ToHead4Edges.forEach(edge => {
        const fromId = edge.from || edge.source;
        const toId = edge.to || edge.target;
        console.log(`  ${fromId} -> ${toId} (type: ${edge.type}, weight: ${edge.weight})`);
    });
    
    // Check for flow_to_new_port edges specifically
    const flowToNewPortEdges = data.edges.filter(e => e.type === 'flow_to_new_port');
    console.log(`flow_to_new_port edges total: ${flowToNewPortEdges.length}`);
    
    if (flowToNewPortEdges.length > 0) {
        console.log('Sample flow_to_new_port edge:', flowToNewPortEdges[0]);
    }
}

function closeNodeDetails() {
    const detailsPanel = document.getElementById('nodeDetails');
    if (detailsPanel) {
        detailsPanel.style.display = 'none';
    }
}