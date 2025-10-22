#!/usr/bin/env python3
"""
Startup script for Perfusion Chatbot Interface
This script sets up and launches the chatbot server with proper error handling
"""

import os
import sys
import subprocess
import time

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['flask', 'flask_socketio', 'torch', 'numpy', 'matplotlib']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Please install missing packages:")
        print("   pip install -r requirements_chatbot.txt")
        return False
    
    return True

def check_trained_models():
    """Check if trained DQN models exist"""
    output_dir = os.path.expanduser("~/Desktop/Simulator/New_System_Results")
    
    eye_models = [
        os.path.join(output_dir, 'best_dqn_agent_EYE.pth'),
        os.path.join(output_dir, 'final_dqn_agent_EYE.pth')
    ]
    
    vca_models = [
        os.path.join(output_dir, 'best_dqn_agent_VCA.pth'),
        os.path.join(output_dir, 'final_dqn_agent_VCA.pth')
    ]
    
    eye_available = any(os.path.exists(model) for model in eye_models)
    vca_available = any(os.path.exists(model) for model in vca_models)
    
    if not eye_available and not vca_available:
        print("❌ No trained DQN models found!")
        print(f"   Expected location: {output_dir}")
        print("   Please train a DQN agent first using dqn_new_system.py")
        return False
    
    if eye_available:
        print("✅ EYE scenario model found")
    else:
        print("⚠️  EYE scenario model not found")
        
    if vca_available:
        print("✅ VCA scenario model found") 
    else:
        print("⚠️  VCA scenario model not found")
    
    return True

def main():
    """Main startup function"""
    print("🏥 Perfusion Chatbot Interface Startup")
    print("=" * 50)
    
    # Check current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"📁 Current directory: {current_dir}")
    
    # Check requirements
    print("\n🔍 Checking requirements...")
    if not check_requirements():
        return 1
    print("✅ All required packages are installed")
    
    # Check trained models
    print("\n🤖 Checking for trained models...")
    if not check_trained_models():
        return 1
    
    # Check if config and init files exist
    config_path = os.path.join(current_dir, 'config.py')
    init_path = os.path.join(current_dir, 'init.py')
    operations_path = os.path.join(current_dir, 'operations.py')
    
    missing_files = []
    for file_path, name in [(config_path, 'config.py'), (init_path, 'init.py'), (operations_path, 'operations.py')]:
        if not os.path.exists(file_path):
            missing_files.append(name)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return 1
    
    print("✅ All required files found")
    
    # Create directories if they don't exist
    templates_dir = os.path.join(current_dir, 'templates')
    if not os.path.exists(templates_dir):
        print(f"📁 Creating templates directory: {templates_dir}")
        os.makedirs(templates_dir, exist_ok=True)
    
    # Start the server
    print("\n🚀 Starting Perfusion Chatbot Server...")
    print("   Access URL: http://localhost:5001")
    print("   Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Import and run the chatbot server
        from chatbot_server import socketio, app
        socketio.run(app, host='0.0.0.0', port=5001, debug=False)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
