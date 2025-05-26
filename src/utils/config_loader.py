"""
Configuration loader utility for the AI model development project.
"""
import os
import yaml
from typing import Dict, Any

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file. If None, uses the default config.yaml.
        
    Returns:
        Dictionary containing the configuration.
    """
    if config_path is None:
        # Get the project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        config_path = os.path.join(project_root, "config.yaml")
    
    # Check if the config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    # Load the configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def get_data_paths(config: Dict[str, Any] = None) -> Dict[str, str]:
    """
    Get the data paths from the configuration.
    
    Args:
        config: Configuration dictionary. If None, loads the configuration.
        
    Returns:
        Dictionary containing the data paths.
    """
    if config is None:
        config = load_config()
    
    # Get the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    # Get the data paths from the configuration
    train_path = os.path.join(project_root, config['data']['train_path'])
    test_path = os.path.join(project_root, config['data']['test_path'])
    
    return {
        'train_path': train_path,
        'test_path': test_path
    }

def get_model_params(model_name: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get the parameters for a specific model from the configuration.
    
    Args:
        model_name: Name of the model (e.g., 'random_forest', 'xgboost').
        config: Configuration dictionary. If None, loads the configuration.
        
    Returns:
        Dictionary containing the model parameters.
    """
    if config is None:
        config = load_config()
    
    if model_name not in config['models']:
        raise ValueError(f"Model {model_name} not found in configuration")
    
    return config['models'][model_name]

if __name__ == "__main__":
    # Test the configuration loader
    config = load_config()
    print("Configuration loaded successfully:")
    print(f"Data settings: {config['data']}")
    print(f"Model settings: {list(config['models'].keys())}")
