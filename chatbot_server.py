#!/usr/bin/env python3
"""
Chatbot Server for Perfusion Process Monitoring
Real-time display of DQN agent evaluation with hour-by-hour updates
"""

import os
import sys
import json
import time
import asyncio
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import numpy as np
import torch

# Add the new_system directory to path for imports
sys.path.append(os.path.dirname(__file__))
import config
import init
from dqn_new_system import (
    NewSimulationEnv, DQNAgent, load_agent
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'perfusion_chatbot_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables to store simulation state
simulation_running = False
current_episode_data = None
agent = None
env = None

def format_parameter_message(param_name, value, thresholds=None, hour=0):
    """Format parameter information as a chatbot message"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Determine status based on thresholds
    status = "normal"
    status_emoji = "✅"
    warning_msg = ""
    
    if thresholds:
        critical_low, low, high, critical_high = thresholds
        if value <= critical_low or value >= critical_high:
            status = "critical"
            status_emoji = "🚨"
            warning_msg = " - CRITICAL LEVEL!"
        elif value <= low or value >= high:
            status = "warning"
            status_emoji = "⚠️"
            warning_msg = " - Warning level"
    
    message = f"**Hour {hour}** - {param_name}: {value:.2f}{warning_msg}"
    
    return {
        "timestamp": timestamp,
        "hour": hour,
        "parameter": param_name,
        "value": value,
        "status": status,
        "message": message,
        "emoji": status_emoji
    }

def format_action_message(action_name, action_value, hour=0):
    """Format action information as a chatbot message"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    action_descriptions = {
        -1: "decrease",
        0: "maintain",
        1: "increase"
    }
    
    action_emojis = {
        -1: "⬇️",
        0: "➡️", 
        1: "⬆️"
    }
    
    action_desc = action_descriptions.get(int(action_value), "maintain")
    action_emoji = action_emojis.get(int(action_value), "➡️")
    
    message = f"**Action** - {action_name}: {action_desc} {action_emoji}"
    
    return {
        "timestamp": timestamp,
        "hour": hour,
        "action": action_name,
        "value": int(action_value),
        "description": action_desc,
        "message": message,
        "emoji": action_emoji
    }

def get_parameter_thresholds(scenario, param_idx):
    """Get thresholds for a parameter based on scenario"""
    try:
        if param_idx < len(config.criticalDepletion):
            return [
                config.criticalDepletion[param_idx],
                config.depletion[param_idx], 
                config.excess[param_idx],
                config.criticalExcess[param_idx]
            ]
    except:
        pass
    return None

@app.route('/')
def index():
    """Serve the chatbot interface"""
    try:
        return render_template('chatbot.html')
    except Exception as e:
        print(f"ERROR loading template: {e}")
        return f"Error loading template: {e}", 500

@app.route('/debug')
def debug():
    """Serve the debug page"""
    try:
        with open(os.path.join(os.path.dirname(__file__), 'debug_socketio.html'), 'r') as f:
            return f.read()
    except Exception as e:
        print(f"ERROR loading debug page: {e}")
        return f"Error loading debug page: {e}", 500

@app.route('/api/test_message', methods=['POST'])
def test_message():
    """Send a test message via SocketIO"""
    try:
        data = request.get_json() or {}
        message = data.get('message', 'Test message from server')
        
        print(f"DEBUG: Sending test message: {message}")
        socketio.emit('chat_message', {
            "type": "system",  
            "message": message,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        print("DEBUG: Test message sent successfully")
        
        return jsonify({"success": True, "message": "Test message sent"})
    except Exception as e:
        print(f"ERROR sending test message: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/start_simulation', methods=['POST'])
def start_simulation():
    """Start a new simulation episode"""
    global simulation_running, agent, env
    
    if simulation_running:
        return jsonify({"error": "Simulation already running"}), 400
    
    try:
        # Get scenario from request or use default
        data = request.get_json() or {}
        scenario = data.get('scenario', init.get_scenario_type())
        
        # Initialize environment
        env = NewSimulationEnv(scenario=scenario)
        
        # Load agent
        output_dir = os.path.expanduser("~/Desktop/Simulator/New_System_Results")
        best_agent_path = os.path.join(output_dir, f'best_dqn_agent_{scenario}.pth')
        final_agent_path = os.path.join(output_dir, f'final_dqn_agent_{scenario}.pth')
        
        if os.path.exists(best_agent_path):
            agent = load_agent(best_agent_path)
        elif os.path.exists(final_agent_path):
            agent = load_agent(final_agent_path)
        else:
            return jsonify({"error": f"No trained agent found for {scenario} scenario"}), 404
        
        if agent is None:
            return jsonify({"error": "Failed to load agent"}), 500
        
        # Start simulation in background
        simulation_running = True
        print(f"DEBUG: About to start background task for {scenario}")
        socketio.start_background_task(run_simulation_with_updates, scenario)
        print(f"DEBUG: Background task started")
        
        return jsonify({
            "message": f"Starting {scenario} perfusion simulation...",
            "scenario": scenario
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stop_simulation', methods=['POST'])
def stop_simulation():
    """Stop the current simulation"""
    global simulation_running
    simulation_running = False
    return jsonify({"message": "Simulation stopped"})

@app.route('/api/status')
def get_status():
    """Get current simulation status"""
    return jsonify({
        "running": simulation_running,
        "scenario": env.scenario if env else None
    })

def run_simulation_with_updates(scenario):
    """Run simulation with real-time chatbot updates"""
    global simulation_running, current_episode_data, agent, env
    
    print(f"DEBUG: Starting simulation for scenario: {scenario}")
    print(f"DEBUG: simulation_running = {simulation_running}")
    print(f"DEBUG: agent is None: {agent is None}")
    print(f"DEBUG: env is None: {env is None}")
    
    try:
        # Send initial message
        print("DEBUG: About to send initial message")
        socketio.emit('chat_message', {
            "type": "system",
            "message": f"🏥 Starting {scenario} perfusion simulation...",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        print("DEBUG: Initial message sent")
        
        # Send a test message to verify SocketIO is working in background task
        time.sleep(1)
        socketio.emit('chat_message', {
            "type": "info",
            "message": f"🧪 Test message from background task - {datetime.now().strftime('%H:%M:%S')}",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        print("DEBUG: Test message sent")
        
        # Reset environment
        state = env.reset()
        
        # Get parameter names based on scenario
        if scenario == "EYE":
            param_names = ["Temperature", "VR", "pH", "pvO2", "Glucose", "Insulin"]
            param_indices = [0, 3, 4, 6, 9, 10]
        else:  # VCA
            param_names = ["Temperature", "VR", "pH", "pvO2", "Glucose", "Insulin"]
            param_indices = [0, 3, 4, 6, 9, 10]
        
        action_names = ["Temp", "Press", "FiO2", "Glucose", "Insulin", "Bicarb", "Vasodil", "Dial_In", "Dial_Out"]
        
        # Send initial state
        socketio.emit('chat_message', {
            "type": "system", 
            "message": f"📊 **Initial State (Hour 0)**",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        # Display initial parameters
        for i, (param_name, param_idx) in enumerate(zip(param_names, param_indices)):
            value = env.big_state[param_idx]
            thresholds = get_parameter_thresholds(scenario, param_idx)
            param_msg = format_parameter_message(param_name, value, thresholds, 0)
            
            socketio.emit('chat_message', {
                "type": "parameter",
                "data": param_msg,
                "message": param_msg["message"],
                "timestamp": param_msg["timestamp"]
            })
        
        # Set agent to evaluation mode
        agent.policy_net.eval()
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0  # No exploration during evaluation
        
        # Run episode
        total_reward = 0
        step_count = 0
        max_steps = 24
        
        socketio.emit('chat_message', {
            "type": "system",
            "message": "🚀 Beginning perfusion monitoring...",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        done = False
        while not done and step_count < max_steps and simulation_running:
            # Add delay for realistic monitoring
            time.sleep(2)  # 2 second delay between hours
            
            if not simulation_running:
                break
                
            # Choose action
            action = agent.choose_action(state)
            action_decoded = env.decode_action(action)
            
            # Take step
            next_state, reward, done, info = env.step(action, train=False)
            
            step_count += 1
            total_reward += reward
            hours_survived = info.get("hours_survived", step_count)
            
            # Send hour update
            socketio.emit('chat_message', {
                "type": "system",
                "message": f"⏰ **Hour {int(hours_survived)}**",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            # Send action information
            socketio.emit('chat_message', {
                "type": "system", 
                "message": "🎯 **AI Decision - Actions Taken:**",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            for i, (action_name, action_value) in enumerate(zip(action_names, action_decoded)):
                if i < len(action_decoded) and action_value != 0:  # Only show non-zero actions
                    action_msg = format_action_message(action_name, action_value, int(hours_survived))
                    socketio.emit('chat_message', {
                        "type": "action",
                        "data": action_msg,
                        "message": action_msg["message"],
                        "timestamp": action_msg["timestamp"]
                    })
            
            # Send updated parameters
            socketio.emit('chat_message', {
                "type": "system",
                "message": "📈 **Updated Parameters:**",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            for i, (param_name, param_idx) in enumerate(zip(param_names, param_indices)):
                value = env.big_state[param_idx]
                thresholds = get_parameter_thresholds(scenario, param_idx)
                param_msg = format_parameter_message(param_name, value, thresholds, int(hours_survived))
                
                socketio.emit('chat_message', {
                    "type": "parameter",
                    "data": param_msg,
                    "message": param_msg["message"],
                    "timestamp": param_msg["timestamp"]
                })
            
            # Check for critical conditions
            if info.get("critical_failure", False):
                socketio.emit('chat_message', {
                    "type": "warning",
                    "message": "⚠️ **CRITICAL CONDITION DETECTED!**",
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
            
            # Send reward information
            if reward > 0:
                socketio.emit('chat_message', {
                    "type": "info",
                    "message": f"💚 Episode reward: +{reward:.1f} (Total: {total_reward:.1f})",
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
            
            state = next_state
            
            # Check if episode is done
            if done:
                if hours_survived >= 24:
                    socketio.emit('chat_message', {
                        "type": "success",
                        "message": f"🎉 **SUCCESS!** Perfusion completed successfully! Survived {hours_survived:.1f} hours.",
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                else:
                    socketio.emit('chat_message', {
                        "type": "error", 
                        "message": f"💔 **Episode ended early** at {hours_survived:.1f} hours. Total reward: {total_reward:.1f}",
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                break
        
        # Restore agent epsilon
        agent.epsilon = original_epsilon
        
        # Send final summary
        socketio.emit('chat_message', {
            "type": "system",
            "message": f"📋 **Final Summary:**\n- Duration: {hours_survived:.1f} hours\n- Total Reward: {total_reward:.1f}\n- Status: {'Success' if hours_survived >= 24 else 'Early termination'}",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
    except Exception as e:
        print(f"ERROR in simulation: {e}")
        import traceback
        traceback.print_exc()
        
        socketio.emit('chat_message', {
            "type": "error",
            "message": f"❌ **Error during simulation:** {str(e)}",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
    
    finally:
        print("DEBUG: Simulation finishing, setting simulation_running to False")
        simulation_running = False
        socketio.emit('simulation_complete', {
            "message": "Simulation completed",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('DEBUG: Client connected')
    try:
        emit('chat_message', {
            "type": "system",
            "message": "🤖 Welcome to the Perfusion Monitoring System! Click 'Start Simulation' to begin.",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        print('DEBUG: Welcome message sent')
    except Exception as e:
        print(f'ERROR sending welcome message: {e}')

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('DEBUG: Client disconnected')
    global simulation_running
    # Stop simulation if client disconnects
    if simulation_running:
        print('DEBUG: Stopping simulation due to client disconnect')
        simulation_running = False

if __name__ == '__main__':
    try:
        # Create templates directory if it doesn't exist
        templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        os.makedirs(templates_dir, exist_ok=True)
        os.makedirs(static_dir, exist_ok=True)
        
        print("Starting Perfusion Chatbot Server...")
        print("Access the interface at: http://localhost:5001")
        
        # Test template loading
        template_path = os.path.join(templates_dir, 'chatbot.html')
        if not os.path.exists(template_path):
            print(f"ERROR: Template not found at {template_path}")
        else:
            print(f"Template found at {template_path}")
        
        # Run the Flask-SocketIO application
        socketio.run(app, host='0.0.0.0', port=5001, debug=False)
        
    except Exception as e:
        print(f"ERROR starting server: {e}")
        import traceback
        traceback.print_exc()
