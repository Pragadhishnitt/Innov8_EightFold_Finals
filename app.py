import subprocess
import sys
import webbrowser
import time
import os
import signal

def cleanup_ports():
    """Kill any existing processes on port 8000 before starting service"""
    print("🧹 Cleaning up any processes on port 8000...")
    try:
        # Kill processes on port 8000
        subprocess.run(["fuser", "-k", "8000/tcp"], check=False, stderr=subprocess.DEVNULL)
        
        # Kill lingering Python processes by name
        subprocess.run(["pkill", "-f", "main.py"], check=False, stderr=subprocess.DEVNULL)
        
        time.sleep(2)  # Allow ports to release
        print("✅ Port 8000 cleaned up.")
    except Exception as e:
        print(f"⚠️ Warning: Could not clean ports: {e}")

def run_main_service():
    """Run the main CodeSage service on port 8000"""
    print("🚀 Starting CodeSage service on port 8000...")
    process = subprocess.Popen([sys.executable, "main.py"])
    return process

def check_service_health(port, service_name):
    """Check if a service is running on the given port"""
    try:
        import requests
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ {service_name} is healthy on port {port}")
            return True
    except Exception as e:
        print(f"⏳ {service_name} not ready yet...")
        return False

def wait_for_service():
    """Wait for the main service to be ready"""
    print("⏳ Waiting for service to start...")
    
    main_ready = False
    max_attempts = 30
    attempts = 0
    
    while not main_ready and attempts < max_attempts:
        time.sleep(2)
        attempts += 1
        
        main_ready = check_service_health(8000, "Main Service")
        
        if attempts % 5 == 0:
            print(f"⏳ Still waiting... (attempt {attempts}/{max_attempts})")
    
    if main_ready:
        print("🎉 Service is ready!")
        return True
    else:
        print("❌ Service failed to start properly")
        return False

def open_interface():
    """Open web interface"""
    print("🌐 Opening web interface...")
    
    # Open main service
    webbrowser.open('http://localhost:8000/')
    
    print("📱 Web interface opened: http://localhost:8000/")

def print_service_info():
    """Print information about the running service"""
    print("\n" + "="*60)
    print("🎯 CODESAGE AI INTERVIEWER - RUNNING")
    print("="*60)
    print("\n📋 Available Endpoints:")
    print("   🏠 Home: http://localhost:8000/")
    print("   📝 Interview: http://localhost:8000/interview")
    print("   📊 Monitor: http://localhost:8000/monitor")
    print("   🔍 Health: http://localhost:8000/health")
    print("   📚 API Docs: http://localhost:8000/docs")
    print("\n🔧 Features:")
    print("   • Real-time code analysis and execution")
    print("   • Live audio transcription with Gemini AI")
    print("   • WebSocket communication")
    print("   • Database persistence")
    print("   • Integrated audio processing")
    print("\n⚡ WebSocket:")
    print("   • ws://localhost:8000/ws/{session_id}")
    print("\n📊 Monitoring:")
    print("   • Active sessions tracking")
    print("   • Audio processing status")
    print("   • System health metrics")
    print("\n" + "="*60)

def handle_shutdown(main_process):
    """Gracefully shutdown the service"""
    print("\n🛑 Shutting down service...")
    
    try:
        if main_process and main_process.poll() is None:
            print("   ⏳ Sending termination signal...")
            main_process.terminate()
            
            # Wait for graceful shutdown
            try:
                main_process.wait(timeout=10)
                print("   ✅ Service stopped gracefully")
            except subprocess.TimeoutExpired:
                print("   ⚠️ Forcing shutdown...")
                main_process.kill()
                main_process.wait()
                print("   ✅ Service force stopped")
        
    except Exception as e:
        print(f"   ⚠️ Error during shutdown: {e}")
    
    print("👋 Service stopped. Goodbye!")

def main():
    """Main function to run the CodeSage service"""
    print("🎬 Initializing CodeSage AI Interviewer...")
    
    cleanup_ports()
    
    main_process = None
    
    try:
        main_process = run_main_service()
        
        service_ready = wait_for_service()
        
        if not service_ready:
            print("❌ Failed to start service properly")
            print("💡 Try checking the logs or running 'python main.py' directly to see errors")
            return
        
        open_interface()
        print_service_info()
        
        print("\n🎯 System Status: RUNNING")
        print("   Press Ctrl+C to stop the service")
        print("\n📊 Live Status:")
        
        # Monitor the process
        while True:
            time.sleep(10)
            
            if main_process.poll() is not None:
                print("⚠️ Service stopped unexpectedly")
                print("💡 Check the console output above for error details")
                break
            
            # Optional: Print a heartbeat
            print("💓 Service running...", end="\r")
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("💡 Try running 'python main.py' directly to see detailed errors")
    
    finally:
        if main_process:
            handle_shutdown(main_process)

def test_dependencies():
    """Test if all required dependencies are available"""
    required_files = ['main.py', 'database_manager.py']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("   Please ensure all required files are in the current directory")
        return False
    
    try:
        # Test critical imports
        import fastapi, uvicorn, psutil
        print("✅ Core dependencies available")
        
        # Test optional imports
        optional_missing = []
        try:
            import supabase
        except ImportError:
            optional_missing.append("supabase")
        
        try:
            import google.generativeai
        except ImportError:
            optional_missing.append("google-generativeai")
        
        if optional_missing:
            print(f"⚠️ Optional dependencies missing: {optional_missing}")
            print("   Some features may be limited")
        else:
            print("✅ All optional dependencies available")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing critical dependency: {e}")
        print("   Please install required packages with:")
        print("   pip install fastapi uvicorn psutil")
        return False

def check_environment():
    """Check if environment is properly configured"""
    print("🔍 Checking environment...")
    
    # Check for .env file
    if os.path.exists('.env'):
        print("✅ .env file found")
    else:
        print("⚠️ .env file not found - some features may not work")
    
    # Check templates directory
    if os.path.exists('templates'):
        template_files = ['home.html', 'index.html', 'monitor.html']
        missing_templates = [f for f in template_files if not os.path.exists(f'templates/{f}')]
        
        if missing_templates:
            print(f"⚠️ Missing template files: {missing_templates}")
        else:
            print("✅ All template files found")
    else:
        print("❌ Templates directory not found")
        return False
    
    return True

if __name__ == "__main__":
    print("🎯 CodeSage AI Interviewer Launcher")
    print("=" * 40)
    
    if not test_dependencies():
        print("\n💡 To install dependencies:")
        print("   pip install fastapi uvicorn psutil python-dotenv")
        print("   pip install supabase google-generativeai  # Optional")
        sys.exit(1)
    
    if not check_environment():
        print("\n💡 Make sure you have:")
        print("   - templates/ directory with HTML files")
        print("   - .env file with your API keys")
        sys.exit(1)
    
    print("✅ Environment check passed")
    print("\n🚀 Starting CodeSage...")
    main()