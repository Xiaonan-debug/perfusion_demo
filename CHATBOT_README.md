# Perfusion Monitoring Chatbot Interface

A real-time web-based chatbot interface for monitoring DQN agent evaluation in perfusion simulations. This interface provides hour-by-hour updates of the perfusion process in a conversational format.

## Features

- 🤖 **Real-time Chatbot Interface**: Interactive chatbot that displays simulation progress
- 📊 **Hour-by-hour Monitoring**: Live updates of perfusion parameters every hour
- 🎯 **AI Decision Tracking**: Shows the AI agent's decisions and actions
- ⚠️ **Critical Alert System**: Immediate notifications for critical conditions
- 📈 **Live Statistics**: Real-time statistics and performance metrics
- 🏥 **Multi-scenario Support**: Supports both EYE and VCA perfusion scenarios

## Requirements

- Python 3.7+
- Trained DQN models (from `dqn_new_system.py`)
- Flask and SocketIO dependencies

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_chatbot.txt
```

### 2. Ensure Trained Models Exist

Make sure you have trained DQN models in the `New_System_Results` directory:
- `best_dqn_agent_EYE.pth` or `final_dqn_agent_EYE.pth` for Eye scenario
- `best_dqn_agent_VCA.pth` or `final_dqn_agent_VCA.pth` for VCA scenario

### 3. Start the Chatbot Server

```bash
python start_chatbot.py
```

Or directly:

```bash
python chatbot_server.py
```

### 4. Access the Interface

Open your web browser and go to: `http://localhost:5000`

## How to Use

1. **Select Scenario**: Choose between EYE or VCA perfusion scenarios
2. **Start Simulation**: Click "Start Simulation" to begin monitoring
3. **Watch Live Updates**: The chatbot will provide real-time updates including:
   - Initial parameter values
   - AI decisions and actions taken each hour
   - Updated parameter values after each intervention
   - Critical alerts and warnings
   - Final results and success/failure status

4. **Monitor Statistics**: The control panel shows:
   - Current simulation hour
   - Total reward accumulated
   - Number of messages received
   - System status

## Interface Components

### Chat Messages

Different types of messages are color-coded:

- **🔵 System Messages**: General information and hour updates
- **🟢 Parameter Updates**: Current values of physiological parameters
- **🟡 Actions**: AI agent decisions and interventions
- **🟠 Warnings**: Threshold violations and concerning trends
- **🔴 Critical Alerts**: Emergency conditions requiring immediate attention
- **✅ Success**: Successful completion of 24-hour perfusion

### Control Panel

- **Scenario Selection**: Choose simulation type
- **Start/Stop Controls**: Control simulation execution
- **Live Statistics**: Real-time performance metrics
- **Status Indicator**: Current system status

## Technical Details

### Backend Architecture

- **Flask Server**: Web server handling HTTP requests
- **SocketIO**: WebSocket communication for real-time updates
- **DQN Integration**: Direct integration with trained DQN agents
- **Simulation Wrapper**: Real-time evaluation with streaming updates

### Frontend Features

- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Updates**: WebSocket-based live communication
- **Modern UI**: Clean, medical-grade interface design
- **Auto-scrolling**: Automatic scrolling to latest messages

## Customization

### Adding New Parameters

To monitor additional parameters, modify the `param_indices` and `param_names` lists in `chatbot_server.py`:

```python
# For EYE scenario
param_names = ["Temperature", "VR", "pH", "pvO2", "Glucose", "Insulin", "NewParam"]
param_indices = [0, 3, 4, 6, 9, 10, 15]  # Add new index
```

### Modifying Thresholds

Update the threshold values in `config.py` to change when warnings and alerts are triggered.

### Customizing Messages

Modify the message formatting functions in `chatbot_server.py`:
- `format_parameter_message()`
- `format_action_message()`

## Troubleshooting

### Common Issues

1. **"No trained agent found"**
   - Ensure DQN models exist in `New_System_Results` directory
   - Train models using `dqn_new_system.py` first

2. **"Missing required packages"**
   - Install dependencies: `pip install -r requirements_chatbot.txt`

3. **"Connection failed"**
   - Check if port 5000 is available
   - Ensure no firewall is blocking the connection

4. **"Simulation not starting"**
   - Check that `config.py`, `init.py`, and `operations.py` exist
   - Verify DQN model files are not corrupted

### Debug Mode

For debugging, modify `chatbot_server.py` and set `debug=True`:

```python
socketio.run(app, host='0.0.0.0', port=5000, debug=True)
```

## Files Structure

```
new_system/
├── chatbot_server.py          # Flask backend server
├── start_chatbot.py           # Startup script
├── requirements_chatbot.txt   # Python dependencies
├── CHATBOT_README.md          # This file
└── templates/
    └── chatbot.html           # Frontend interface
```

## Performance Notes

- Each simulation hour has a 2-second delay for realistic monitoring
- The interface can handle multiple concurrent connections
- Memory usage scales with message history (automatically managed)
- Real-time updates may have slight delays based on network conditions

## Future Enhancements

- Historical simulation playback
- Parameter trend analysis and predictions
- Multiple scenario comparison
- Export functionality for simulation reports
- Advanced filtering and search capabilities
