"""
Utility script to set up the directory structure for the predictive maintenance project.
"""
import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.config_loader import load_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_directories():
    """
    Create the necessary directories for the project.
    """
    logger.info("Setting up directory structure...")
    
    # Get the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    # Define directories to create
    directories = [
        os.path.join(project_root, "data", "raw"),
        os.path.join(project_root, "data", "processed"),
        os.path.join(project_root, "data", "splits"),
        os.path.join(project_root, "models"),
        os.path.join(project_root, "results"),
        os.path.join(project_root, "logs")
    ]
    
    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    logger.info("Directory structure setup complete!")

if __name__ == "__main__":
    setup_directories()
