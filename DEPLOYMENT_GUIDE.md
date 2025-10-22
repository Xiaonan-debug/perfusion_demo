# 🌐 Online Perfusion Chatbot Deployment Guide

Deploy your AI-powered perfusion monitoring system online for global access!

## 🚀 Quick Deploy Options

### Option 1: Railway (Recommended - Easiest)

**Railway** is perfect for this Flask + SocketIO application:

1. **Create Railway Account**: Go to [railway.app](https://railway.app) and sign up
2. **Connect GitHub**: Link your GitHub account
3. **Create New Project**: Click "New Project" → "Deploy from GitHub repo"
4. **Upload Files**: Create a new GitHub repo with these files:
   ```
   new_system/
   ├── online_perfusion_chatbot.py
   ├── requirements_deployment.txt
   ├── Procfile
   ├── runtime.txt
   └── DEPLOYMENT_GUIDE.md
   ```
5. **Deploy**: Railway will automatically detect and deploy your Flask app
6. **Access**: Get your live URL (e.g., `https://your-app.railway.app`)

### Option 2: Render (Free Tier Available)

1. **Create Render Account**: Go to [render.com](https://render.com)
2. **New Web Service**: Click "New" → "Web Service"
3. **Connect Repository**: Connect your GitHub repo
4. **Configure**:
   - Build Command: `pip install -r requirements_deployment.txt`
   - Start Command: `gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT online_perfusion_chatbot:app`
5. **Deploy**: Render will build and deploy automatically

### Option 3: Heroku

1. **Create Heroku Account**: Go to [heroku.com](https://heroku.com)
2. **Install Heroku CLI**: Download from heroku.com/cli
3. **Deploy Commands**:
   ```bash
   cd /Users/xluobd/Desktop/Simulator/new_system
   heroku create your-perfusion-app
   git init
   git add .
   git commit -m "Initial deploy"
   git push heroku main
   ```

## 📁 Required Files for Deployment

### 1. `online_perfusion_chatbot.py`
- ✅ Already created - Main application file
- Cloud-optimized version with intelligent AI demo
- No dependency on local trained models
- Built-in realistic perfusion simulation

### 2. `requirements_deployment.txt`
- ✅ Already created - Python dependencies
- Includes Flask, SocketIO, matplotlib, numpy
- Optimized for cloud deployment

### 3. `Procfile`
- ✅ Already created - Tells cloud how to run your app
- Uses gunicorn with eventlet for WebSocket support

### 4. `runtime.txt`
- ✅ Already created - Specifies Python version
- Ensures compatibility across cloud platforms

## 🎯 What Your Online Demo Includes

### **Realistic AI Simulation**
- **Intelligent decision making** that responds to parameter changes
- **Realistic perfusion physics** with proper parameter ranges
- **Smart AI agent** that maintains parameters in safe zones
- **Educational value** showing how AI manages medical systems

### **Professional Interface**  
- **Real-time trajectory charts** updating hour by hour
- **Live parameter monitoring** with safety zone visualization
- **Critical alerts** when parameters become dangerous
- **Performance statistics** tracking AI effectiveness

### **Scenarios Available**
- **Eye Perfusion**: Specialized parameters for ocular perfusion
- **VCA Perfusion**: General vascular perfusion simulation
- **24-hour cycles**: Complete simulation from start to finish

## 🌐 After Deployment

### **Your Online URL**
Once deployed, you'll get a URL like:
- Railway: `https://your-app.railway.app`
- Render: `https://your-app.onrender.com`  
- Heroku: `https://your-app.herokuapp.com`

### **Share Your Demo**
- Send the URL to colleagues, students, or collaborators
- Works on desktop, tablet, and mobile devices
- No installation required - runs in any web browser
- Real-time updates via WebSocket connections

### **Demo Features**
- **Start AI Simulation**: Begins realistic perfusion monitoring
- **Live Charts**: Parameter trajectories build up hour by hour
- **AI Decisions**: See what actions the AI takes and why
- **Critical Management**: Watch AI handle emergency situations
- **Educational Tool**: Perfect for teaching AI in healthcare

## ⚙️ Customization Options

### **Environment Variables**
Set these in your cloud platform's dashboard:
```bash
SECRET_KEY=your_secret_key_here
DEBUG=False
```

### **Parameter Adjustment**
Edit the `DEMO_PARAMS` dictionary in `online_perfusion_chatbot.py` to:
- Modify parameter ranges
- Adjust critical thresholds
- Change AI behavior patterns
- Add new scenarios

### **Styling Changes**
Modify the embedded CSS in the HTML template to:
- Change color schemes
- Adjust layout proportions
- Add your branding
- Customize responsive behavior

## 🔧 Troubleshooting

### **Common Issues**

**1. App Won't Start**
- Check `requirements_deployment.txt` has all dependencies
- Verify `Procfile` command is correct
- Check logs in your cloud platform dashboard

**2. WebSocket Not Working**
- Ensure eventlet is installed (`pip install eventlet`)
- Check that your cloud platform supports WebSockets
- Verify CORS settings allow your domain

**3. Charts Not Loading**
- Check matplotlib is installed
- Verify the `/api/trajectory_chart` endpoint works
- Check browser console for JavaScript errors

### **Performance Optimization**
- **Memory**: Charts are generated in memory and cached
- **CPU**: Simulation runs in background threads
- **Network**: WebSocket connections are efficient
- **Scaling**: Can handle multiple concurrent users

## 📊 Monitoring Your Deployment

### **Usage Analytics**
Most cloud platforms provide:
- **Request counts**: How many people are using your demo
- **Response times**: How fast your app responds
- **Error rates**: Any issues that need fixing
- **Resource usage**: Memory and CPU consumption

### **Logs Access**
View real-time logs to see:
- User connections and disconnections
- Simulation start/stop events
- Any errors or warnings
- Performance metrics

## 🎓 Educational Use

### **Perfect For**
- **Medical AI courses**: Demonstrate AI in healthcare
- **Engineering classes**: Show real-time systems
- **Research presentations**: Interactive AI demonstrations
- **Conference demos**: Professional medical AI showcase

### **Key Learning Points**
- **AI Decision Making**: How AI analyzes and responds to data
- **Real-time Monitoring**: Critical systems management
- **Parameter Control**: Maintaining complex system stability
- **Medical Applications**: AI in healthcare scenarios

---

## 🚀 Ready to Deploy?

1. **Choose a platform** (Railway recommended for beginners)
2. **Upload your files** to a GitHub repository
3. **Connect and deploy** using the platform's interface
4. **Share your URL** and start demonstrating AI-powered perfusion monitoring!

Your intelligent perfusion monitoring system will be available worldwide! 🌍🏥🤖
