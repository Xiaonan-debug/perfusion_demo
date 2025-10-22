#!/usr/bin/env python3
"""
Online Perfusion Chatbot - Cloud Deployment Version
Real-time display of DQN agent evaluation with hour-by-hour updates
Optimized for cloud deployment (Heroku, Railway, Render, etc.)
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for cloud
import matplotlib.pyplot as plt
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import io
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'online_perfusion_chatbot_key')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Global state
simulation_running = False

# Online demo trajectory data (since we may not have trained models in cloud)
trajectory_data = {
    'hours': [],
    'parameters': {},
    'actions': [],
    'rewards': [],
    'scenario': None,
    'param_names': [],
    'param_indices': []
}

# Demo perfusion parameters and realistic ranges
DEMO_PARAMS = {
    'EYE': {
        'names': ["Temperature", "VR", "pH", "pvO2", "Glucose", "Insulin"],
        'initial': [37.0, 2.5, 7.35, 300, 6.0, 16.0],
        'ranges': [(35.5, 38.5), (1.0, 5.0), (7.0, 7.6), (100, 500), (3.0, 12.0), (5.0, 40.0)],
        'critical_low': [35.0, 0.5, 6.9, 80, 2.0, 3.0],
        'critical_high': [39.0, 8.0, 7.7, 600, 15.0, 60.0],
        'warning_low': [36.0, 1.0, 7.1, 120, 3.5, 8.0],
        'warning_high': [38.0, 6.0, 7.5, 450, 10.0, 35.0]
    },
    'VCA': {
        'names': ["Temperature", "VR", "pH", "pvO2", "Glucose", "Insulin"], 
        'initial': [36.5, 1.8, 7.40, 250, 5.5, 160.0],
        'ranges': [(35.0, 39.0), (0.8, 3.5), (7.0, 7.6), (80, 400), (3.0, 10.0), (80.0, 250.0)],
        'critical_low': [34.5, 0.3, 6.8, 60, 2.5, 50.0],
        'critical_high': [40.0, 5.0, 7.8, 500, 12.0, 300.0],
        'warning_low': [35.5, 0.8, 7.0, 100, 3.5, 100.0],
        'warning_high': [38.5, 3.0, 7.6, 350, 8.0, 220.0]
    }
}

@app.route('/')
def index():
    """Serve the online chatbot interface"""
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Online Perfusion Monitoring System</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
        .container { max-width: 100vw; margin: 0; padding: 15px; height: 100vh; display: flex; flex-direction: column; }
        .header { background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); color: white; padding: 15px 25px; border-radius: 10px; text-align: center; margin-bottom: 15px; }
        .header h1 { margin: 0 0 5px 0; font-size: 1.5rem; font-weight: 600; }
        .header p { margin: 0; opacity: 0.9; font-size: 0.9rem; }
        .main-content { display: flex; gap: 15px; flex: 1; min-height: 0; }
        .chart-main-area { flex: 1; background: white; border-radius: 15px; padding: 20px; overflow: hidden; box-shadow: 0 8px 32px rgba(0,0,0,0.1); display: flex; flex-direction: column; }
        .chart-header { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 15px; text-align: center; font-weight: 600; font-size: 1.2rem; border-radius: 10px; margin-bottom: 15px; }
        .chart-container-main { flex: 1; background: #f8f9fa; border-radius: 10px; padding: 15px; display: flex; align-items: center; justify-content: center; min-height: 400px; }
        .sidebar { width: 380px; display: flex; flex-direction: column; gap: 12px; }
        .chat-area { background: white; border-radius: 15px; padding: 0; overflow: hidden; box-shadow: 0 8px 32px rgba(0,0,0,0.1); flex: 2; min-height: 300px; display: flex; flex-direction: column; }
        .chat-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 10px; text-align: center; font-weight: 600; font-size: 0.95rem; }
        .messages { flex: 1; overflow-y: auto; padding: 12px; background: #f8f9fa; min-height: 250px; }
        .message { margin: 10px 0; padding: 10px; border-radius: 8px; line-height: 1.3; animation: fadeIn 0.3s ease-in; font-size: 0.85rem; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .system { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
        .parameter { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; }
        .action { background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%); color: #2d3436; }
        .info { background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); color: white; }
        .success { background: linear-gradient(135deg, #00b894 0%, #00cec9 100%); color: white; border-left: 4px solid #00b894; }
        .error { background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%); color: white; border-left: 4px solid #d63031; }
        .warning { background: linear-gradient(135deg, #fd79a8 0%, #fdcb6e 100%); color: #2d3436; border-left: 4px solid #e17055; }
        .timestamp { font-size: 0.7rem; opacity: 0.6; margin-top: 4px; }
        .control-panel { background: white; border-radius: 12px; padding: 15px; box-shadow: 0 6px 24px rgba(0,0,0,0.1); flex: none; }
        .control-panel h3 { color: #2d3436; margin: 0 0 10px 0; font-size: 1rem; }
        .status { padding: 6px 12px; border-radius: 15px; font-size: 0.75rem; font-weight: 600; text-align: center; margin-bottom: 10px; }
        .connected { background: linear-gradient(135deg, #00b894 0%, #00cec9 100%); color: white; }
        .disconnected { background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%); color: white; }
        .running { background: linear-gradient(135deg, #00b894 0%, #00cec9 100%); color: white; animation: pulse 2s infinite; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.7; } 100% { opacity: 1; } }
        .control-group { margin-bottom: 10px; }
        .control-group label { display: block; margin-bottom: 4px; font-weight: 500; color: #636e72; font-size: 0.8rem; }
        select, button { width: 100%; padding: 6px; border: 1px solid #ddd; border-radius: 5px; font-size: 0.85rem; transition: all 0.3s ease; }
        select:focus { outline: none; border-color: #4facfe; box-shadow: 0 0 0 2px rgba(79, 172, 254, 0.1); }
        button { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; border: none; cursor: pointer; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; margin-bottom: 6px; }
        button:hover { transform: translateY(-1px); box-shadow: 0 2px 10px rgba(79, 172, 254, 0.4); }
        button:disabled { background: #95a5a6; cursor: not-allowed; transform: none; box-shadow: none; }
        .stop-btn { background: linear-gradient(135deg, #e17055 0%, #d63031 100%); }
        .stop-btn:hover { box-shadow: 0 2px 10px rgba(214, 48, 49, 0.4); }
        .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 10px 0; }
        .stat-card { background: #f8f9fa; padding: 8px; border-radius: 5px; text-align: center; border: 1px solid #e9ecef; }
        .stat-value { font-size: 1.2rem; font-weight: 700; color: #2d3436; margin-bottom: 2px; }
        .stat-label { font-size: 0.65rem; color: #636e72; text-transform: uppercase; letter-spacing: 0.3px; }
        .control-info { margin-top: 10px; padding: 8px; background: #f8f9fa; border-radius: 5px; font-size: 0.7rem; color: #666; line-height: 1.3; }
        .messages::-webkit-scrollbar { width: 6px; }
        .messages::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 10px; }
        .messages::-webkit-scrollbar-thumb { background: #888; border-radius: 10px; }
        .demo-badge { background: linear-gradient(135deg, #fd79a8 0%, #fdcb6e 100%); color: #2d3436; padding: 4px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: 600; display: inline-block; margin-left: 10px; }
        
        /* Responsive design */
        @media (max-width: 1200px) {
            .main-content { flex-direction: column; }
            .sidebar { width: 100%; flex-direction: row; gap: 12px; }
            .control-panel { flex: 0 0 300px; }
            .chat-area { flex: 1; min-height: 250px; }
            .chart-main-area { min-height: 450px; }
        }
        
        @media (max-width: 768px) {
            .container { padding: 8px; }
            .sidebar { flex-direction: column; width: 100%; }
            .control-panel { flex: none; }
            .chat-area { min-height: 200px; }
            .chart-main-area { min-height: 350px; }
            .stats-grid { grid-template-columns: repeat(4, 1fr); gap: 6px; }
            .stat-card { padding: 6px; }
            .stat-value { font-size: 1rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 Online Perfusion Monitoring System</h1>
            <p>Live AI-powered perfusion simulation with real-time trajectory analysis</p>
            <span class="demo-badge">🌐 ONLINE DEMO</span>
        </div>
        
        <div class="main-content">
            <!-- Main Chart Area -->
            <div class="chart-main-area">
                <div class="chart-header">📊 Real-Time Parameter Trajectories - AI Agent Performance</div>
                <div class="chart-container-main">
                    <div id="chartContainerMain" style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; color: #666; font-size: 1.1rem;">
                        <span>🌐 Online Demo Ready!<br><br>Select a scenario and click "Start AI Simulation" to begin monitoring</span>
                    </div>
                </div>
            </div>
            
            <!-- Sidebar with Controls and Messages -->
            <div class="sidebar">
                <!-- Control Panel -->
                <div class="control-panel">
                    <h3>🤖 AI Control Panel</h3>
                    
                    <div id="status" class="status disconnected">Status: Connecting...</div>
                    
                    <div class="control-group">
                        <label for="scenario">Perfusion Scenario:</label>
                        <select id="scenario">
                            <option value="EYE">Eye Perfusion</option>
                            <option value="VCA">VCA Perfusion</option>
                        </select>
                    </div>
                    
                    <div class="control-group">
                        <button onclick="startOnlineSimulation()">🚀 Start AI Simulation</button>
                        <button class="stop-btn" onclick="stopSimulation()">⏹️ Stop Simulation</button>
                    </div>
                    
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-value" id="currentHour">0</div>
                            <div class="stat-label">Current Hour</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="totalReward">0</div>
                            <div class="stat-label">Total Reward</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="messageCount">0</div>
                            <div class="stat-label">Messages</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="agentActions">0</div>
                            <div class="stat-label">AI Actions</div>
                        </div>
                    </div>
                    
                    <div class="control-info">
                        <strong>Online AI Demo:</strong><br>
                        • Realistic perfusion physics<br>
                        • Intelligent AI decision making<br>
                        • Real-time parameter tracking<br>
                        • 24-hour simulation cycles
                    </div>
                </div>
                
                <!-- Chat Messages Area -->
                <div class="chat-area">
                    <div class="chat-header">💬 Live AI Monitoring Feed</div>
                    <div id="messages" class="messages"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const messagesDiv = document.getElementById('messages');
        const statusDiv = document.getElementById('status');
        
        let messageCount = 0;
        let currentHour = 0;
        let totalReward = 0;
        let agentActions = 0;
        
        // Initialize SocketIO
        console.log('Connecting to online AI perfusion monitoring system...');
        const socket = io();
        
        socket.on('connect', function() {
            console.log('Connected to AI system');
            statusDiv.className = 'status connected';
            statusDiv.textContent = '✅ Connected to AI System';
            addMessage('🟢 Connected to Online AI Perfusion System!', 'system');
        });
        
        socket.on('disconnect', function() {
            console.log('Disconnected from AI system');
            statusDiv.className = 'status disconnected';
            statusDiv.textContent = '❌ Disconnected from AI System';
            addMessage('🔴 Disconnected from AI system', 'error');
        });
        
        socket.on('chat_message', function(data) {
            console.log('Received AI data:', data);
            addMessage(data.message, data.type);
            
            // Update statistics based on real data
            if (data.type === 'system' && data.message.includes('Hour')) {
                const hourMatch = data.message.match(/Hour (\\d+)/);
                if (hourMatch) {
                    currentHour = parseInt(hourMatch[1]);
                    document.getElementById('currentHour').textContent = currentHour;
                }
            }
            
            if (data.type === 'info' && data.message.includes('reward')) {
                const rewardMatch = data.message.match(/reward[:\\s]+([\\d.-]+)/i);
                if (rewardMatch) {
                    totalReward = parseFloat(rewardMatch[1]);
                    document.getElementById('totalReward').textContent = Math.round(totalReward);
                }
            }
            
            if (data.type === 'action') {
                agentActions++;
                document.getElementById('agentActions').textContent = agentActions;
            }
        });
        
        socket.on('simulation_complete', function(data) {
            console.log('AI simulation complete:', data);
            addMessage('🎉 Online AI evaluation completed!', 'success');
            statusDiv.className = 'status connected';
            statusDiv.textContent = '✅ Simulation Complete';
        });
        
        socket.on('chart_update', function(data) {
            console.log('Received chart update');
            updateChart();
        });
        
        function addMessage(message, type = 'info') {
            const div = document.createElement('div');
            div.className = `message ${type}`;
            
            // Format bold text
            const formattedMessage = message.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>');
            
            div.innerHTML = `
                <div>${formattedMessage}</div>
                <div class="timestamp">${new Date().toLocaleTimeString()}</div>
            `;
            messagesDiv.appendChild(div);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            
            messageCount++;
            document.getElementById('messageCount').textContent = messageCount;
        }
        
        function startOnlineSimulation() {
            console.log('Starting online AI simulation...');
            const scenario = document.getElementById('scenario').value;
            
            statusDiv.className = 'status running';
            statusDiv.textContent = '🚀 Starting AI Simulation...';
            
            // Reset statistics
            currentHour = 0;
            totalReward = 0;
            agentActions = 0;
            document.getElementById('currentHour').textContent = '0';
            document.getElementById('totalReward').textContent = '0';
            document.getElementById('agentActions').textContent = '0';
            
            // Reset chart
            resetChart();
            
            fetch('/api/start_online', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ scenario: scenario })
            })
            .then(response => response.json())
            .then(data => {
                console.log('Online start response:', data);
                if (data.error) {
                    addMessage('❌ Error: ' + data.error, 'error');
                    statusDiv.className = 'status connected';
                    statusDiv.textContent = '❌ Error Starting';
                } else {
                    statusDiv.className = 'status running';
                    statusDiv.textContent = '🔄 AI Agent Running';
                }
            })
            .catch(error => {
                console.error('Start error:', error);
                addMessage('❌ Failed to start: ' + error.message, 'error');
                statusDiv.className = 'status connected';
                statusDiv.textContent = '❌ Start Failed';
            });
        }
        
        function stopSimulation() {
            console.log('Stopping online simulation...');
            fetch('/api/stop', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                console.log('Stop response:', data);
                addMessage('⏹️ Simulation stopped by user', 'warning');
                statusDiv.className = 'status connected';
                statusDiv.textContent = '⏹️ Stopped';
            })
            .catch(error => {
                console.error('Stop error:', error);
            });
        }
        
        function updateChart() {
            const chartContainer = document.getElementById('chartContainerMain');
            const timestamp = new Date().getTime();
            
            // Update chart with current trajectory data
            chartContainer.innerHTML = `
                <img src="/api/trajectory_chart?t=${timestamp}" 
                     style="max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);"
                     onerror="this.style.display='none'; this.parentElement.innerHTML='<span style=\\"color: #999;\\">Loading chart...</span>'"
                     onload="console.log('Chart updated successfully')">
            `;
        }
        
        function resetChart() {
            const chartContainer = document.getElementById('chartContainerMain');
            chartContainer.innerHTML = '<span style="text-align: center;">🌐 Online Demo Ready!<br><br>Select a scenario and click "Start AI Simulation" to begin monitoring</span>';
        }
    </script>
</body>
</html>
    '''

def generate_demo_trajectory_chart():
    """Generate demo trajectory chart for online deployment"""
    global trajectory_data
    
    if not trajectory_data['hours'] or not trajectory_data['parameters']:
        return None
    
    try:
        # Set up the plot with professional styling for online display
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Professional color palette
        agent_color = '#2E86DE'  # Blue for agent trajectory
        critical_color = '#E74C3C'  # Red for critical thresholds
        warning_color = '#F39C12'  # Orange for warning thresholds
        safe_zone_color = '#D5F4E6'  # Light green for safe zone
        warning_zone_color = '#FCF3CF'  # Light yellow for warning
        danger_zone_color = '#FADBD8'  # Light red for danger
        
        hours = trajectory_data['hours']
        scenario = trajectory_data['scenario']
        param_names = trajectory_data['param_names']
        
        for i, param_name in enumerate(param_names):
            if i < len(axes) and param_name in trajectory_data['parameters']:
                ax = axes[i]
                
                # Get parameter values
                values = trajectory_data['parameters'][param_name]
                
                # Get demo thresholds for safety zones
                demo_params = DEMO_PARAMS[scenario]
                if param_name in demo_params['names']:
                    idx = demo_params['names'].index(param_name)
                    critical_low = demo_params['critical_low'][idx]
                    critical_high = demo_params['critical_high'][idx]
                    warning_low = demo_params['warning_low'][idx]
                    warning_high = demo_params['warning_high'][idx]
                    
                    if len(values) > 0:
                        # Calculate plot limits with some margin
                        y_min = min(critical_low * 0.9, min(values) * 0.95)
                        y_max = max(critical_high * 1.1, max(values) * 1.05)
                        
                        # Draw safety zones
                        ax.axhspan(y_min, critical_low, alpha=0.15, color=danger_zone_color, zorder=0)
                        ax.axhspan(critical_high, y_max, alpha=0.15, color=danger_zone_color, zorder=0)
                        ax.axhspan(critical_low, warning_low, alpha=0.1, color=warning_zone_color, zorder=0)
                        ax.axhspan(warning_high, critical_high, alpha=0.1, color=warning_zone_color, zorder=0)
                        ax.axhspan(warning_low, warning_high, alpha=0.12, color=safe_zone_color, zorder=0)
                        
                        # Draw threshold lines
                        ax.axhline(y=critical_low, color=critical_color, linestyle='--', 
                                  linewidth=2, alpha=0.8, zorder=1, label='Critical')
                        ax.axhline(y=critical_high, color=critical_color, linestyle='--', 
                                  linewidth=2, alpha=0.8, zorder=1)
                        ax.axhline(y=warning_low, color=warning_color, linestyle=':', 
                                  linewidth=1.5, alpha=0.7, zorder=1, label='Warning')
                        ax.axhline(y=warning_high, color=warning_color, linestyle=':', 
                                  linewidth=1.5, alpha=0.7, zorder=1)
                
                # Plot trajectory with professional styling
                if len(hours) > 1:
                    # Plot line
                    ax.plot(hours, values, color=agent_color, linewidth=3, 
                           marker='o', markersize=6, markerfacecolor='white', 
                           markeredgewidth=2, markeredgecolor=agent_color, 
                           label='AI Agent', zorder=4)
                elif len(hours) == 1:
                    # Single point
                    ax.plot(hours[0], values[0], color=agent_color, marker='o', 
                           markersize=8, markerfacecolor='white', markeredgewidth=2, 
                           markeredgecolor=agent_color, label='AI Agent', zorder=4)
                
                # Styling with larger fonts for main display
                ax.set_title(f'{param_name}', fontsize=14, fontweight='bold', pad=12)
                ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
                ax.set_ylabel('Value', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.6)
                ax.set_axisbelow(True)
                ax.tick_params(labelsize=11)
                
                # Set reasonable axis limits
                if len(values) > 0 and param_name in demo_params['names']:
                    ax.set_ylim(y_min, y_max)
                elif len(values) > 0:
                    margin = (max(values) - min(values)) * 0.1 if len(values) > 1 else 1
                    ax.set_ylim(min(values) - margin, max(values) + margin)
                
                # Set x-axis to show up to 24 hours
                ax.set_xlim(0, max(24, max(hours) + 1) if hours else 24)
                
                # Add legend for first subplot only
                if i == 0:
                    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        
        # Hide unused subplots
        for i in range(len(param_names), len(axes)):
            axes[i].set_visible(False)
        
        # Add main title for large display
        current_hour = max(hours) if hours else 0
        fig.suptitle(f'🌐 Online {scenario} AI Demo - Hour {current_hour}/24 🤖', 
                    fontsize=16, fontweight='bold', y=0.98, color='#2C3E50')
        
        # Tight layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save to memory buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        img_buffer.seek(0)
        
        plt.close(fig)  # Important: close figure to free memory
        
        return img_buffer
        
    except Exception as e:
        print(f"Error generating demo chart: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')  # Clean up any open figures
        return None

@app.route('/api/trajectory_chart')
def serve_trajectory_chart():
    """Serve the current trajectory chart for online demo"""
    img_buffer = generate_demo_trajectory_chart()
    
    if img_buffer:
        return send_file(img_buffer, mimetype='image/png', as_attachment=False)
    else:
        # Return a placeholder image if no data
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, '🌐 Online Demo\nStart simulation to see real-time charts', 
                ha='center', va='center', fontsize=14, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1) 
        ax.axis('off')
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close(fig)
        
        return send_file(img_buffer, mimetype='image/png', as_attachment=False)

@app.route('/api/start_online', methods=['POST'])
def start_online_simulation():
    """Start online demo simulation"""
    global simulation_running, trajectory_data
    
    print("=== STARTING ONLINE AI DEMO ===")
    
    if simulation_running:
        return {"error": "Online simulation already running"}, 400
    
    try:
        data = request.get_json() or {}
        scenario = data.get('scenario', 'EYE')
        print(f"Starting online demo for scenario: {scenario}")
        
        # Reset trajectory data for new simulation
        trajectory_data = {
            'hours': [],
            'parameters': {},
            'actions': [],
            'rewards': [],
            'scenario': scenario,
            'param_names': DEMO_PARAMS[scenario]['names'],
            'param_indices': []
        }
        
        # Initialize parameter dictionaries
        for param_name in trajectory_data['param_names']:
            trajectory_data['parameters'][param_name] = []
        
        # Start demo simulation
        simulation_running = True
        socketio.start_background_task(online_demo_simulation, scenario)
        
        return {"success": True, "message": f"Starting online {scenario} AI demo"}
        
    except Exception as e:
        print(f"ERROR starting online simulation: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}, 500

@app.route('/api/stop', methods=['POST'])
def stop_simulation():
    """Stop the online simulation"""
    global simulation_running
    print("=== STOPPING ONLINE SIMULATION ===")
    simulation_running = False
    return {"success": True, "message": "Online simulation stopped"}

def online_demo_simulation(scenario):
    """Run realistic online demo simulation"""
    global simulation_running, trajectory_data
    
    print(f"=== ONLINE AI DEMO STARTED for {scenario} ===")
    
    try:
        # Send initial message
        socketio.emit('chat_message', {
            "type": "system",
            "message": f"🌐 **Starting Online {scenario} AI Demo**",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        demo_params = DEMO_PARAMS[scenario]
        param_names = demo_params['names']
        
        # Initialize with demo parameters
        current_values = demo_params['initial'].copy()
        
        # Add initial data point (Hour 0)
        trajectory_data['hours'].append(0)
        trajectory_data['rewards'].append(0)
        
        # Send initial state
        socketio.emit('chat_message', {
            "type": "system",
            "message": f"📊 **Initial {scenario} Parameters (Hour 0)**",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        # Display initial parameters and collect trajectory data
        for i, param_name in enumerate(param_names):
            value = current_values[i]
            trajectory_data['parameters'][param_name].append(value)
            
            socketio.emit('chat_message', {
                "type": "parameter",
                "message": f"**{param_name}**: {value:.2f}",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
        
        # Send initial chart update
        socketio.emit('chart_update', {
            "hour": 0,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        socketio.emit('chat_message', {
            "type": "system",
            "message": "🤖 **AI Agent Starting Intelligent Decision Making**",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        # Run realistic demo simulation for up to 24 hours
        total_reward = 0
        step_count = 0
        max_steps = 24
        
        while step_count < max_steps and simulation_running:
            if not simulation_running:
                print("Online demo stopped by user")
                break
            
            # Add realistic delay for monitoring
            time.sleep(2)  # 2 seconds per hour
            
            step_count += 1
            hours_survived = step_count
            
            # Simulate realistic AI decision making
            ai_actions = simulate_intelligent_decisions(current_values, demo_params, hours_survived)
            
            # Apply AI decisions to update parameters
            current_values = apply_ai_decisions(current_values, ai_actions, demo_params)
            
            # Calculate realistic reward
            reward = calculate_demo_reward(current_values, demo_params, hours_survived)
            total_reward += reward
            
            print(f"Online demo - Hour {hours_survived}: reward={reward:.2f}, total={total_reward:.2f}")
            
            # Send hour update
            socketio.emit('chat_message', {
                "type": "system",
                "message": f"⏰ **Hour {hours_survived}** - AI Agent Analysis",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            # Send AI actions
            action_descriptions = []
            for i, (param_name, action) in enumerate(zip(param_names, ai_actions)):
                if action != 0:
                    action_desc = "optimize" if action > 0 else "stabilize"
                    action_descriptions.append(f"{param_name}: {action_desc}")
            
            if action_descriptions:
                socketio.emit('chat_message', {
                    "type": "action",
                    "message": f"🎯 **AI Decisions**: {', '.join(action_descriptions)}",
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
            else:
                socketio.emit('chat_message', {
                    "type": "action",
                    "message": f"🎯 **AI Decision**: Maintain optimal parameters",
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
            
            # Add current hour data to trajectory
            trajectory_data['hours'].append(hours_survived)
            trajectory_data['rewards'].append(total_reward)
            trajectory_data['actions'].append(ai_actions.copy())
            
            # Send updated parameters
            socketio.emit('chat_message', {
                "type": "system",
                "message": f"📈 **Updated Parameters (Hour {hours_survived})**",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            for i, param_name in enumerate(param_names):
                value = current_values[i]
                
                # Store parameter value in trajectory data
                trajectory_data['parameters'][param_name].append(value)
                
                # Check for critical values using demo thresholds
                status = ""
                if value <= demo_params['critical_low'][i] or value >= demo_params['critical_high'][i]:
                    status = " ⚠️ CRITICAL"
                elif value <= demo_params['warning_low'][i] or value >= demo_params['warning_high'][i]:
                    status = " ⚠️ Warning"
                
                msg_type = "warning" if "CRITICAL" in status else "parameter"
                
                socketio.emit('chat_message', {
                    "type": msg_type,
                    "message": f"**{param_name}**: {value:.2f}{status}",
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
            
            # Send chart update after parameter update
            socketio.emit('chart_update', {
                "hour": hours_survived,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            # Send reward information
            if reward != 0:
                socketio.emit('chat_message', {
                    "type": "info",
                    "message": f"💰 **AI Performance**: {reward:.1f} (Total: {total_reward:.1f})",
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
            
            # Check for critical conditions
            critical_failure = any(v <= demo_params['critical_low'][i] or v >= demo_params['critical_high'][i] 
                                 for i, v in enumerate(current_values))
            
            if critical_failure:
                socketio.emit('chat_message', {
                    "type": "error",
                    "message": "🚨 **CRITICAL CONDITION!** AI agent working to stabilize parameters.",
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                break
            
            # Check for success
            if hours_survived >= 24:
                socketio.emit('chat_message', {
                    "type": "success",
                    "message": f"🎉 **SUCCESS!** Online {scenario} AI demo completed! Successfully managed perfusion for 24 hours.",
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                break
        
        # Send final summary
        success = hours_survived >= 24 and not critical_failure
        socketio.emit('chat_message', {
            "type": "system",
            "message": f"📋 **Online AI Demo Summary:**\n- Scenario: {scenario}\n- Duration: {hours_survived} hours\n- Total Performance Score: {total_reward:.1f}\n- Status: {'Success' if success else 'Learning experience'}\n- AI Agent: {'Excellent performance' if success else 'Continuous improvement'}",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        # Send completion event
        socketio.emit('simulation_complete', {
            "message": f"Online {scenario} AI demo completed",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
    
    except Exception as e:
        print(f"ERROR in online demo: {e}")
        import traceback
        traceback.print_exc()
        
        socketio.emit('chat_message', {
            "type": "error",
            "message": f"❌ **Error in online demo:** {str(e)}",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
    
    finally:
        print("=== ONLINE AI DEMO FINISHED ===")
        simulation_running = False

def simulate_intelligent_decisions(current_values, demo_params, hour):
    """Simulate realistic AI decision making"""
    actions = [0] * len(current_values)
    
    for i, (current, target_low, target_high) in enumerate(zip(
        current_values, demo_params['warning_low'], demo_params['warning_high']
    )):
        # Add some intelligent behavior with slight randomness
        if current < target_low:
            actions[i] = 1  # Increase
        elif current > target_high:
            actions[i] = -1  # Decrease
        else:
            # Small random adjustments in safe zone
            if np.random.random() < 0.3:  # 30% chance of adjustment
                actions[i] = np.random.choice([-1, 0, 1])
    
    return actions

def apply_ai_decisions(current_values, ai_actions, demo_params):
    """Apply AI decisions to update parameter values realistically"""
    new_values = current_values.copy()
    
    for i, (current, action) in enumerate(zip(current_values, ai_actions)):
        # Apply action with realistic physics
        if action == 1:  # Increase
            change = np.random.uniform(0.05, 0.3) * abs(demo_params['ranges'][i][1] - demo_params['ranges'][i][0]) * 0.1
            new_values[i] = min(current + change, demo_params['ranges'][i][1])
        elif action == -1:  # Decrease
            change = np.random.uniform(0.05, 0.3) * abs(demo_params['ranges'][i][1] - demo_params['ranges'][i][0]) * 0.1
            new_values[i] = max(current - change, demo_params['ranges'][i][0])
        else:
            # Natural drift with small random variation
            drift = np.random.uniform(-0.02, 0.02) * abs(demo_params['ranges'][i][1] - demo_params['ranges'][i][0]) * 0.05
            new_values[i] = np.clip(current + drift, demo_params['ranges'][i][0], demo_params['ranges'][i][1])
    
    return new_values

def calculate_demo_reward(current_values, demo_params, hour):
    """Calculate realistic reward based on how well parameters are maintained"""
    reward = 0
    
    for i, (current, warn_low, warn_high, crit_low, crit_high) in enumerate(zip(
        current_values, demo_params['warning_low'], demo_params['warning_high'],
        demo_params['critical_low'], demo_params['critical_high']
    )):
        if crit_low <= current <= crit_high:
            if warn_low <= current <= warn_high:
                reward += 2  # In safe zone
            else:
                reward += 1  # In warning zone
        else:
            reward -= 3  # In critical zone
    
    # Bonus for survival
    if hour >= 12:
        reward += 5
    if hour >= 20:
        reward += 10
    
    return reward

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('=== CLIENT CONNECTED TO ONLINE AI DEMO ===')
    emit('chat_message', {
        "type": "system",
        "message": "🌐 **Connected to Online AI Demo!** This is a realistic simulation of AI-powered perfusion management. Ready to demonstrate intelligent medical AI!",
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('=== CLIENT DISCONNECTED FROM ONLINE AI DEMO ===')
    global simulation_running
    if simulation_running:
        print('Stopping online demo due to client disconnect')
        simulation_running = False

# Health check endpoint for cloud platforms
@app.route('/health')
def health_check():
    return {"status": "healthy", "service": "online_perfusion_chatbot"}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5005))
    print("="*60)
    print("🌐 STARTING ONLINE AI PERFUSION MONITORING SYSTEM")
    print("="*60)
    print(f"Port: {port}")
    print("This is the cloud-ready version with intelligent demo!")
    print("Ready for deployment to Heroku, Railway, Render, etc.")
    print("="*60)
    
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
