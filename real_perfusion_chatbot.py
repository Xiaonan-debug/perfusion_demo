#!/usr/bin/env python3
"""
Real Perfusion Chatbot - Connected to DQN System
Shows actual DQN agent evaluation with real perfusion data
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import io
import base64

# Add the new_system directory to path for imports
sys.path.append(os.path.dirname(__file__))
import config
import init
from operations import single_step
from dqn_new_system import NewSimulationEnv, load_agent

app = Flask(__name__)
app.config['SECRET_KEY'] = 'real_perfusion_chatbot_key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
simulation_running = False
agent = None
env = None

# Real-time plotting data
trajectory_data = {
    'hours': [],
    'parameters': {},
    'actions': [],
    'rewards': [],
    'scenario': None,
    'param_names': [],
    'param_indices': []
}

@app.route('/')
def index():
    """Serve the chatbot interface"""
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real Perfusion Monitoring System</title>
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
            <h1>🏥 Real Perfusion Monitoring System</h1>
            <p>Live DQN Agent Evaluation with Hour-by-Hour Perfusion Updates</p>
        </div>
        
        <div class="main-content">
            <!-- Main Chart Area -->
            <div class="chart-main-area">
                <div class="chart-header">📊 Real-Time Parameter Trajectories - DQN Agent Performance</div>
                <div class="chart-container-main">
                    <div id="chartContainerMain" style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; color: #666; font-size: 1.1rem;">
                        <span>🏥 Charts will appear when simulation starts...<br><br>Select a scenario and click "Start Real DQN Evaluation" to begin monitoring</span>
                    </div>
                </div>
            </div>
            
            <!-- Sidebar with Controls and Messages -->
            <div class="sidebar">
                <!-- Control Panel -->
                <div class="control-panel">
                    <h3>⚙️ DQN Control Panel</h3>
                    
                    <div id="status" class="status disconnected">Status: Connecting...</div>
                    
                    <div class="control-group">
                        <label for="scenario">Perfusion Scenario:</label>
                        <select id="scenario">
                            <option value="EYE">Eye Perfusion</option>
                            <option value="VCA">VCA Perfusion</option>
                        </select>
                    </div>
                    
                    <div class="control-group">
                        <button onclick="startRealSimulation()">🚀 Start Real DQN Evaluation</button>
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
                        <strong>Real DQN Integration:</strong><br>
                        • Trained DQN models<br>
                        • Actual perfusion parameters<br>
                        • Real AI decision making<br>
                        • Live 24-hour simulation
                    </div>
                </div>
                
                <!-- Chat Messages Area -->
                <div class="chat-area">
                    <div class="chat-header">💬 Live Monitoring Feed</div>
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
        console.log('Connecting to real perfusion monitoring system...');
        const socket = io();
        
        socket.on('connect', function() {
            console.log('Connected to DQN system');
            statusDiv.className = 'status connected';
            statusDiv.textContent = '✅ Connected to DQN System';
            addMessage('🟢 Connected to Real DQN Perfusion System!', 'system');
        });
        
        socket.on('disconnect', function() {
            console.log('Disconnected from DQN system');
            statusDiv.className = 'status disconnected';
            statusDiv.textContent = '❌ Disconnected from DQN System';
            addMessage('🔴 Disconnected from DQN system', 'error');
        });
        
        socket.on('chat_message', function(data) {
            console.log('Received real data:', data);
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
            console.log('Real simulation complete:', data);
            addMessage('🎉 Real DQN evaluation completed!', 'success');
            statusDiv.className = 'status connected';
            statusDiv.textContent = '✅ Evaluation Complete';
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
        
        function startRealSimulation() {
            console.log('Starting real DQN evaluation...');
            const scenario = document.getElementById('scenario').value;
            
            statusDiv.className = 'status running';
            statusDiv.textContent = '🚀 Starting DQN Evaluation...';
            
            // Reset statistics
            currentHour = 0;
            totalReward = 0;
            agentActions = 0;
            document.getElementById('currentHour').textContent = '0';
            document.getElementById('totalReward').textContent = '0';
            document.getElementById('agentActions').textContent = '0';
            
            // Reset chart
            resetChart();
            
            fetch('/api/start_real', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ scenario: scenario })
            })
            .then(response => response.json())
            .then(data => {
                console.log('Real start response:', data);
                if (data.error) {
                    addMessage('❌ Error: ' + data.error, 'error');
                    statusDiv.className = 'status connected';
                    statusDiv.textContent = '❌ Error Starting';
                } else {
                    statusDiv.className = 'status running';
                    statusDiv.textContent = '🔄 DQN Agent Running';
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
            console.log('Stopping real simulation...');
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
            chartContainer.innerHTML = '<span style="text-align: center;">🏥 Charts will appear when simulation starts...<br><br>Select a scenario and click "Start Real DQN Evaluation" to begin monitoring</span>';
        }
    </script>
</body>
</html>
    '''

def get_thresholds(scenario, param_idx):
    """Get threshold values for plotting safety zones"""
    try:
        if param_idx < len(config.criticalDepletion):
            return [
                config.criticalDepletion[param_idx],  # Critical low
                config.depletion[param_idx],          # Warning low  
                config.excess[param_idx],             # Warning high
                config.criticalExcess[param_idx]      # Critical high
            ]
    except:
        pass
    return None

def generate_trajectory_chart():
    """Generate real-time trajectory chart showing parameter evolution"""
    global trajectory_data
    
    if not trajectory_data['hours'] or not trajectory_data['parameters']:
        return None
    
    try:
        # Set up the plot with professional styling for main display
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
        param_indices = trajectory_data['param_indices']
        
        for i, (param_name, param_idx) in enumerate(zip(param_names, param_indices)):
            if i < len(axes) and param_name in trajectory_data['parameters']:
                ax = axes[i]
                
                # Get parameter values
                values = trajectory_data['parameters'][param_name]
                
                # Get thresholds for safety zones
                thresholds = get_thresholds(scenario, param_idx)
                
                if thresholds and len(values) > 0:
                    critical_low, warning_low, warning_high, critical_high = thresholds
                    
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
                           label='DQN Agent', zorder=4)
                elif len(hours) == 1:
                    # Single point
                    ax.plot(hours[0], values[0], color=agent_color, marker='o', 
                           markersize=8, markerfacecolor='white', markeredgewidth=2, 
                           markeredgecolor=agent_color, label='DQN Agent', zorder=4)
                
                # Styling with larger fonts for main display
                ax.set_title(f'{param_name}', fontsize=14, fontweight='bold', pad=12)
                ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
                ax.set_ylabel('Value', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.6)
                ax.set_axisbelow(True)
                ax.tick_params(labelsize=11)
                
                # Set reasonable axis limits
                if len(values) > 0:
                    if thresholds:
                        ax.set_ylim(y_min, y_max)
                    else:
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
        fig.suptitle(f'{scenario} Perfusion Scenario - DQN Agent Performance (Hour {current_hour}/24)', 
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
        print(f"Error generating trajectory chart: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')  # Clean up any open figures
        return None

@app.route('/api/trajectory_chart')
def serve_trajectory_chart():
    """Serve the current trajectory chart"""
    img_buffer = generate_trajectory_chart()
    
    if img_buffer:
        return send_file(img_buffer, mimetype='image/png', as_attachment=False)
    else:
        # Return a placeholder image if no data
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No trajectory data available\nStart simulation to see real-time charts', 
                ha='center', va='center', fontsize=12, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1) 
        ax.axis('off')
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close(fig)
        
        return send_file(img_buffer, mimetype='image/png', as_attachment=False)

@app.route('/api/start_real', methods=['POST'])
def start_real_simulation():
    """Start real DQN evaluation with actual perfusion simulation"""
    global simulation_running, agent, env, trajectory_data
    
    print("=== STARTING REAL DQN EVALUATION ===")
    
    if simulation_running:
        return {"error": "Real simulation already running"}, 400
    
    try:
        data = request.get_json() or {}
        scenario = data.get('scenario', 'EYE')
        print(f"Starting real DQN evaluation for scenario: {scenario}")
        
        # Reset trajectory data for new simulation
        trajectory_data = {
            'hours': [],
            'parameters': {},
            'actions': [],
            'rewards': [],
            'scenario': scenario,
            'param_names': [],
            'param_indices': []
        }
        
        # Initialize environment
        env = NewSimulationEnv(scenario=scenario)
        print(f"Environment created for {scenario}")
        
        # Load DQN agent
        output_dir = os.path.expanduser("~/Desktop/Simulator/New_System_Results")
        best_agent_path = os.path.join(output_dir, f'best_dqn_agent_{scenario}.pth')
        final_agent_path = os.path.join(output_dir, f'final_dqn_agent_{scenario}.pth')
        
        if os.path.exists(best_agent_path):
            agent = load_agent(best_agent_path)
            print(f"Loaded best agent: {best_agent_path}")
        elif os.path.exists(final_agent_path):
            agent = load_agent(final_agent_path)
            print(f"Loaded final agent: {final_agent_path}")
        else:
            return {"error": f"No trained DQN agent found for {scenario} scenario"}, 404
        
        if agent is None:
            return {"error": "Failed to load DQN agent"}, 500
        
        # Start real simulation
        simulation_running = True
        socketio.start_background_task(real_dqn_evaluation, scenario)
        
        return {"success": True, "message": f"Starting real {scenario} DQN evaluation"}
        
    except Exception as e:
        print(f"ERROR starting real simulation: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}, 500

@app.route('/api/stop', methods=['POST'])
def stop_simulation():
    """Stop the real simulation"""
    global simulation_running
    print("=== STOPPING REAL SIMULATION ===")
    simulation_running = False
    return {"success": True, "message": "Real simulation stopped"}

def real_dqn_evaluation(scenario):
    """Run real DQN agent evaluation with actual perfusion simulation"""
    global simulation_running, agent, env, trajectory_data
    
    print(f"=== REAL DQN EVALUATION STARTED for {scenario} ===")
    
    try:
        # Send initial message
        socketio.emit('chat_message', {
            "type": "system",
            "message": f"🏥 **Starting Real {scenario} DQN Evaluation**",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        # Parameter names based on scenario
        if scenario == "EYE":
            param_names = ["Temperature", "VR", "pH", "pvO2", "Glucose", "Insulin"]
            param_indices = [0, 3, 4, 6, 9, 10]
        else:  # VCA
            param_names = ["Temperature", "VR", "pH", "pvO2", "Glucose", "Insulin"]
            param_indices = [0, 3, 4, 6, 9, 10]
        
        action_names = ["Temp", "Press", "FiO2", "Glucose", "Insulin", "Bicarb", "Vasodil", "Dial_In", "Dial_Out"]
        
        # Initialize trajectory data
        trajectory_data['param_names'] = param_names
        trajectory_data['param_indices'] = param_indices
        trajectory_data['scenario'] = scenario
        
        # Initialize parameter dictionaries
        for param_name in param_names:
            trajectory_data['parameters'][param_name] = []
        
        # Reset environment with real initial state
        state = env.reset()
        print(f"Environment reset, initial state shape: {state.shape}")
        
        # Send initial state
        socketio.emit('chat_message', {
            "type": "system",
            "message": f"📊 **Initial {scenario} Parameters (Hour 0)**",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        # Add initial data point (Hour 0)
        trajectory_data['hours'].append(0)
        trajectory_data['rewards'].append(0)
        
        # Display initial parameters with real values and collect trajectory data
        for i, (param_name, param_idx) in enumerate(zip(param_names, param_indices)):
            value = env.big_state[param_idx]
            
            # Store initial parameter value
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
        
        # Set agent to evaluation mode
        agent.policy_net.eval()
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0  # No exploration during evaluation
        
        # Run real perfusion simulation
        total_reward = 0
        step_count = 0
        max_steps = 24
        
        socketio.emit('chat_message', {
            "type": "system",
            "message": "🤖 **DQN Agent Starting Decision Making Process**",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        done = False
        while not done and step_count < max_steps and simulation_running:
            if not simulation_running:
                print("Real simulation stopped by user")
                break
            
            # Add realistic delay for monitoring
            time.sleep(2)  # 2 seconds per hour
            
            # Choose action using trained DQN agent
            action = agent.choose_action(state)
            action_decoded = env.decode_action(action)
            
            # Take step in real simulation
            next_state, reward, done, info = env.step(action, train=False)
            
            step_count += 1
            total_reward += reward
            hours_survived = info.get("hours_survived", step_count)
            
            print(f"Real simulation - Hour {int(hours_survived)}: reward={reward:.2f}, total={total_reward:.2f}")
            
            # Send hour update
            socketio.emit('chat_message', {
                "type": "system",
                "message": f"⏰ **Hour {int(hours_survived)}** - DQN Agent Evaluation",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            # Send AI actions (only non-zero actions)
            active_actions = []
            for i, (action_name, action_value) in enumerate(zip(action_names, action_decoded)):
                if i < len(action_decoded) and action_value != 0:
                    action_desc = "increase" if action_value == 1 else "decrease"
                    active_actions.append(f"{action_name}: {action_desc}")
            
            if active_actions:
                socketio.emit('chat_message', {
                    "type": "action",
                    "message": f"🎯 **DQN Actions**: {', '.join(active_actions)}",
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
            else:
                socketio.emit('chat_message', {
                    "type": "action",
                    "message": f"🎯 **DQN Decision**: Maintain all parameters",
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
            
            # Add current hour data to trajectory
            trajectory_data['hours'].append(int(hours_survived))
            trajectory_data['rewards'].append(total_reward)
            trajectory_data['actions'].append(action_decoded.copy())
            
            # Send updated real parameters
            socketio.emit('chat_message', {
                "type": "system",
                "message": f"📈 **Updated Parameters (Hour {int(hours_survived)})**",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            for i, (param_name, param_idx) in enumerate(zip(param_names, param_indices)):
                value = env.big_state[param_idx]
                
                # Store parameter value in trajectory data
                trajectory_data['parameters'][param_name].append(value)
                
                # Check for critical values
                status = ""
                if param_idx < len(config.criticalDepletion):
                    if value <= config.criticalDepletion[param_idx] or value >= config.criticalExcess[param_idx]:
                        status = " ⚠️ CRITICAL"
                    elif value <= config.depletion[param_idx] or value >= config.excess[param_idx]:
                        status = " ⚠️ Warning"
                
                msg_type = "warning" if "CRITICAL" in status else "parameter"
                
                socketio.emit('chat_message', {
                    "type": msg_type,
                    "message": f"**{param_name}**: {value:.2f}{status}",
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
            
            # Send chart update after parameter update
            socketio.emit('chart_update', {
                "hour": int(hours_survived),
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            # Send reward information
            if reward != 0:
                socketio.emit('chat_message', {
                    "type": "info",
                    "message": f"💰 **Reward**: {reward:.1f} (Total: {total_reward:.1f})",
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
            
            # Check for critical conditions
            if info.get("critical_failure", False):
                socketio.emit('chat_message', {
                    "type": "error",
                    "message": "🚨 **CRITICAL CONDITION DETECTED!** Parameter reached dangerous level.",
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
            
            state = next_state
            
            # Check if episode is done
            if done:
                if hours_survived >= 24:
                    socketio.emit('chat_message', {
                        "type": "success",
                        "message": f"🎉 **SUCCESS!** Real {scenario} perfusion completed! Survived {hours_survived:.1f} hours with DQN agent.",
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                else:
                    socketio.emit('chat_message', {
                        "type": "error",
                        "message": f"💔 **Early Termination** - {scenario} simulation ended at {hours_survived:.1f} hours. Total reward: {total_reward:.1f}",
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                break
        
        # Restore agent epsilon
        agent.epsilon = original_epsilon
        
        # Send final summary
        socketio.emit('chat_message', {
            "type": "system",
            "message": f"📋 **Real DQN Evaluation Summary:**\n- Scenario: {scenario}\n- Duration: {hours_survived:.1f} hours\n- Total Reward: {total_reward:.1f}\n- Status: {'Success' if hours_survived >= 24 else 'Early termination'}\n- Agent Performance: {'Excellent' if hours_survived >= 24 else 'Needs improvement'}",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        # Send completion event
        socketio.emit('simulation_complete', {
            "message": f"Real {scenario} DQN evaluation completed",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
    
    except Exception as e:
        print(f"ERROR in real DQN evaluation: {e}")
        import traceback
        traceback.print_exc()
        
        socketio.emit('chat_message', {
            "type": "error",
            "message": f"❌ **Error in real DQN evaluation:** {str(e)}",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
    
    finally:
        print("=== REAL DQN EVALUATION FINISHED ===")
        simulation_running = False

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('=== CLIENT CONNECTED TO REAL DQN SYSTEM ===')
    emit('chat_message', {
        "type": "system",
        "message": "🤖 **Connected to Real DQN Perfusion System!** Ready to evaluate trained agents on actual perfusion simulations.",
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('=== CLIENT DISCONNECTED FROM REAL DQN SYSTEM ===')
    global simulation_running
    if simulation_running:
        print('Stopping real simulation due to client disconnect')
        simulation_running = False

if __name__ == '__main__':
    print("="*60)
    print("🏥 STARTING REAL DQN PERFUSION MONITORING SYSTEM")
    print("="*60)
    print("URL: http://localhost:5005")
    print("This version connects to your trained DQN agents!")
    print("Shows real perfusion simulation data and AI decisions")
    print("Press Ctrl+C to stop")
    print("="*60)
    
    socketio.run(app, host='0.0.0.0', port=5005, debug=True)
