"""
Launch script for Homelessness Prevention App
Run: python run_app.py
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False
    return True

def run_streamlit_app():
    """Launch the Streamlit application"""
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "homelessness_prevention_app.py",
            "--server.port=8501",
            "--server.address=localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

if __name__ == "__main__":
    print("🏠 Homelessness Prevention Support System")
    print("=" * 50)
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        sys.exit(1)
    
    # Install requirements
    print("📦 Installing requirements...")
    if not install_requirements():
        sys.exit(1)
    
    # Run the app
    print("🚀 Starting application...")
    print("📱 Open your browser to: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop")
    
    run_streamlit_app()