"""
Script to run the Predictive Maintenance application.
This script starts both the API server and the Streamlit web UI.
"""
import os
import sys
import subprocess
import time
import threading
import webbrowser
import signal
import logging
import requests
import traceback
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.config_loader import load_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load configuration
try:
    config = load_config()
    # Ensure deployment section exists
    if 'deployment' not in config:
        config['deployment'] = {
            'api_enabled': True,
            'api_port': 8000,
            'ui_port': 8501,
            'save_pipeline': True,
            'pipeline_path': 'models/demo_model.pkl'
        }
    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.error(f"Error loading configuration: {e}")
    # Default configuration
    config = {
        'deployment': {
            'api_enabled': True,
            'api_port': 8000,
            'ui_port': 8501,
            'save_pipeline': True,
            'pipeline_path': 'models/demo_model.pkl'
        }
    }

# Global variables for processes
api_process = None
ui_process = None
stop_event = threading.Event()

def start_api_server():
    """
    Start the FastAPI server.
    
    Returns:
        int: The port the API server is running on.
    """
    global api_process
    
    logger.info("Starting API server...")
    
    # Get the port from config or use default
    port = config['deployment'].get('api_port', 8000)
    
    # Build the command
    cmd = [
        sys.executable,
        "-m", "uvicorn",
        "src.deployment.api:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--reload"
    ]
    
    # Start the process
    try:
        api_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )
        
        logger.info(f"API server started on port {port}")
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        logger.error(traceback.format_exc())
        return None
    
    # Wait for the server to start
    max_retries = 30
    retries = 0
    while retries < max_retries and not stop_event.is_set():
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                logger.info("API server is ready")
                return port
        except requests.exceptions.ConnectionError:
            # This is expected while the server is starting
            pass
        except Exception as e:
            logger.warning(f"Error checking API health: {e}")
        
        retries += 1
        time.sleep(1)
        logger.info(f"Waiting for API server to start... ({retries}/{max_retries})")
    
    if retries == max_retries:
        logger.warning("API server might not be fully ready, but continuing...")
    
    return port

def start_web_ui():
    """
    Start the Streamlit web UI.
    
    Returns:
        int: The port the web UI is running on.
    """
    global ui_process
    
    logger.info("Starting Streamlit web UI...")
    
    # Get the port from config or use default
    port = config['deployment'].get('ui_port', 8501)
    
    # Build the command
    cmd = [
        sys.executable,
        "-m", "streamlit",
        "run",
        os.path.join("src", "deployment", "web_ui.py"),
        "--server.port", str(port)
    ]
    
    # Start the process
    try:
        ui_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )
        
        logger.info(f"Streamlit web UI started on port {port}")
    except Exception as e:
        logger.error(f"Failed to start web UI: {e}")
        logger.error(traceback.format_exc())
        return None
    
    # Wait for the UI to start
    time.sleep(5)
    
    return port

def check_api_health(port):
    """
    Check if the API is healthy.
    
    Args:
        port: API port.
        
    Returns:
        bool: True if API is healthy, False otherwise.
    """
    if port is None:
        return False
        
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def open_browser(api_port, ui_port):
    """
    Open web browser tabs for API docs and web UI.
    """
    if api_port is None or ui_port is None:
        logger.error("Cannot open browser: API or UI port is None")
        return
        
    try:
        # Open API docs
        api_url = f"http://localhost:{api_port}/docs"
        webbrowser.open(api_url)
        
        # Open web UI
        ui_url = f"http://localhost:{ui_port}"
        webbrowser.open(ui_url)
        
        logger.info(f"Opened browser tabs for API docs ({api_url}) and web UI ({ui_url})")
    except Exception as e:
        logger.error(f"Failed to open browser: {e}")

def monitor_processes():
    """
    Monitor the running processes and log their output.
    """
    def log_output(process, name):
        if process is None or process.stdout is None:
            return
            
        try:
            for line in iter(process.stdout.readline, ''):
                if line and not stop_event.is_set():
                    logger.info(f"{name}: {line.strip()}")
                else:
                    break
        except Exception as e:
            logger.error(f"Error monitoring {name} output: {e}")
    
    # Start threads to monitor output
    if api_process and api_process.stdout:
        api_thread = threading.Thread(target=log_output, args=(api_process, "API"))
        api_thread.daemon = True
        api_thread.start()
    
    if ui_process and ui_process.stdout:
        ui_thread = threading.Thread(target=log_output, args=(ui_process, "UI"))
        ui_thread.daemon = True
        ui_thread.start()

def cleanup(signum=None, frame=None):
    """
    Clean up processes on exit.
    """
    logger.info("Shutting down processes...")
    
    # Signal threads to stop
    stop_event.set()
    
    # Terminate API process
    if api_process:
        logger.info("Terminating API server...")
        try:
            api_process.terminate()
            try:
                api_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                api_process.kill()
                logger.info("API server killed")
        except Exception as e:
            logger.error(f"Error terminating API server: {e}")
    
    # Terminate UI process
    if ui_process:
        logger.info("Terminating web UI...")
        try:
            ui_process.terminate()
            try:
                ui_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ui_process.kill()
                logger.info("Web UI killed")
        except Exception as e:
            logger.error(f"Error terminating web UI: {e}")
    
    logger.info("All processes terminated")
    
    # Give time for log messages to be written
    time.sleep(1)

def ensure_model_exists():
    """
    Ensure a model exists for the API to use.
    """
    try:
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
        
        # Create models directory if it doesn't exist
        if not os.path.exists(models_dir):
            os.makedirs(models_dir, exist_ok=True)
            logger.info(f"Created models directory: {models_dir}")
        
        # Check for any model file
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        
        if not model_files:
            logger.info("No model found. Running demo setup...")
            # Import here to avoid circular imports
            from src.utils.demo_setup import setup_demo
            setup_demo()
        else:
            model_path = os.path.join(models_dir, model_files[0])
            logger.info(f"Model found at {model_path}")
    except Exception as e:
        logger.error(f"Error ensuring model exists: {e}")
        logger.error(traceback.format_exc())
        raise

def restart_api_server(port):
    """
    Restart the API server if it has stopped.
    
    Args:
        port: The port to restart the API server on.
        
    Returns:
        int: The port the API server is running on, or None if restart failed.
    """
    global api_process
    
    logger.info("Attempting to restart the API server...")
    
    # Terminate existing process if it's still running
    if api_process and api_process.poll() is None:
        try:
            api_process.terminate()
            api_process.wait(timeout=5)
        except:
            try:
                api_process.kill()
            except:
                pass
    
    # Start a new process
    return start_api_server()

def main():
    """
    Main function to run the application.
    """
    print("=" * 80)
    print(f"  Predictive Maintenance Application - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print("\nStarting application components...\n")
    
    try:
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, cleanup)
        signal.signal(signal.SIGTERM, cleanup)
        
        # Ensure model exists
        ensure_model_exists()
        
        # Start API server
        api_port = start_api_server()
        if api_port is None:
            logger.error("Failed to start API server. Exiting.")
            return
        
        # Wait for API to be healthy
        max_retries = 15
        api_healthy = False
        for i in range(max_retries):
            if check_api_health(api_port):
                logger.info("API is healthy")
                api_healthy = True
                break
            logger.info(f"Waiting for API to be healthy... ({i+1}/{max_retries})")
            time.sleep(2)
        
        if not api_healthy:
            logger.warning("API health check failed. Attempting to restart API...")
            api_port = restart_api_server(api_port)
            if api_port is None:
                logger.error("Failed to restart API server. Exiting.")
                return
                
            # Check health again
            time.sleep(5)
            if not check_api_health(api_port):
                logger.error("API is still not healthy after restart. Exiting.")
                return
        
        # Start web UI
        ui_port = start_web_ui()
        if ui_port is None:
            logger.error("Failed to start web UI. Exiting.")
            return
        
        # Monitor process output
        monitor_processes()
        
        # Open browser tabs
        open_browser(api_port, ui_port)
        
        print("\nApplication is running!")
        print(f"API server: http://localhost:{api_port}")
        print(f"API documentation: http://localhost:{api_port}/docs")
        print(f"Web UI: http://localhost:{ui_port}")
        print("\nPress Ctrl+C to stop the application\n")
        
        # Keep the main thread alive and monitor processes
        api_restart_attempts = 0
        ui_restart_attempts = 0
        max_restart_attempts = 3
        
        while not stop_event.is_set():
            time.sleep(2)
            
            # Check if API process is still running
            if api_process and api_process.poll() is not None:
                logger.error("API server has stopped unexpectedly")
                
                if api_restart_attempts < max_restart_attempts:
                    logger.info(f"Attempting to restart API server (attempt {api_restart_attempts + 1}/{max_restart_attempts})...")
                    api_port = restart_api_server(api_port)
                    api_restart_attempts += 1
                    
                    if api_port is None:
                        logger.error("Failed to restart API server. Exiting.")
                        break
                else:
                    logger.error(f"API server has crashed {max_restart_attempts} times. Giving up.")
                    break
            else:
                # Reset counter if API is running fine
                api_restart_attempts = 0
                
            # Check if UI process is still running
            if ui_process and ui_process.poll() is not None:
                logger.error("Web UI has stopped unexpectedly")
                
                if ui_restart_attempts < max_restart_attempts:
                    logger.info(f"Attempting to restart web UI (attempt {ui_restart_attempts + 1}/{max_restart_attempts})...")
                    ui_port = start_web_ui()
                    ui_restart_attempts += 1
                    
                    if ui_port is None:
                        logger.error("Failed to restart web UI.")
                        # Continue running even if UI fails, as the API might still be useful
                else:
                    logger.error(f"Web UI has crashed {max_restart_attempts} times. Giving up on UI.")
                    # Continue running even if UI fails, as the API might still be useful
            else:
                # Reset counter if UI is running fine
                ui_restart_attempts = 0
    
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt. Shutting down...")
    except Exception as e:
        logger.error(f"Error running application: {e}")
        logger.error(traceback.format_exc())
    finally:
        cleanup()

if __name__ == "__main__":
    main()
