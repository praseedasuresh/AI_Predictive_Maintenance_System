"""
Utility script to set up a demo of the predictive maintenance application.
This script:
1. Generates sample data if needed
2. Trains a simple model
3. Saves the model for the API to use
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.config_loader import load_config
from src.data_processing.preprocessor import DataPreprocessor
from src.data_processing.feature_engineering import FeatureEngineering

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("demo_setup.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_sample_data(num_samples=10000, save_path=None):
    """
    Generate sample data for the predictive maintenance demo.
    
    Args:
        num_samples: Number of samples to generate.
        save_path: Path to save the generated data.
        
    Returns:
        DataFrame with generated data.
    """
    logger.info(f"Generating {num_samples} sample data points...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate machine types
    machine_types = np.random.choice(['L', 'M', 'H'], size=num_samples, p=[0.3, 0.5, 0.2])
    
    # Generate features
    air_temps = np.random.normal(300, 2, num_samples)
    process_temps = np.random.normal(310, 1, num_samples) + 0.5 * (machine_types == 'H').astype(int)
    
    # Rotational speed depends on machine type
    rotational_speeds = np.zeros(num_samples)
    for i, mtype in enumerate(machine_types):
        if mtype == 'L':
            rotational_speeds[i] = np.random.normal(1350, 50, 1)
        elif mtype == 'M':
            rotational_speeds[i] = np.random.normal(1500, 30, 1)
        else:  # 'H'
            rotational_speeds[i] = np.random.normal(1650, 40, 1)
    
    # Generate torque
    torques = np.maximum(30, np.random.normal(40, 5, num_samples) + 2 * (machine_types == 'H').astype(int))
    
    # Generate tool wear
    tool_wears = np.random.uniform(0, 250, num_samples)
    
    # Create feature dataframe
    data = pd.DataFrame({
        'Type': machine_types,
        'Air temperature [K]': air_temps,
        'Process temperature [K]': process_temps,
        'Rotational speed [rpm]': rotational_speeds,
        'Torque [Nm]': torques,
        'Tool wear [min]': tool_wears
    })
    
    # Create derived features to help with failure prediction
    data['Temperature_diff'] = data['Process temperature [K]'] - data['Air temperature [K]']
    data['Power'] = data['Rotational speed [rpm]'] * data['Torque [Nm]'] * (2 * np.pi / 60)
    data['Tool_wear_rate'] = data['Tool wear [min]'] / 250  # Normalized tool wear
    
    # Generate failures based on rules
    # 1. Tool wear failure (TWF)
    twf = (data['Tool wear [min]'] > 200) & (np.random.random(num_samples) < 0.7)
    
    # 2. Heat dissipation failure (HDF)
    hdf = (data['Temperature_diff'] < 8.6) & (data['Rotational speed [rpm]'] < 1380) & (np.random.random(num_samples) < 0.6)
    
    # 3. Power failure (PWF)
    pwf = ((data['Power'] < 3500) | (data['Power'] > 9000)) & (np.random.random(num_samples) < 0.5)
    
    # 4. Overstrain failure (OSF)
    tool_wear_torque = data['Tool wear [min]'] * data['Torque [Nm]']
    thresholds = {'L': 11000, 'M': 12000, 'H': 13000}
    osf = np.zeros(num_samples, dtype=bool)
    for mtype, threshold in thresholds.items():
        mask = (data['Type'] == mtype) & (tool_wear_torque > threshold)
        osf = osf | (mask & (np.random.random(num_samples) < 0.6))
    
    # 5. Random failure (RNF)
    rnf = np.random.random(num_samples) < 0.01
    
    # Combine all failures
    any_failure = twf | hdf | pwf | osf | rnf
    
    # Create failure type column
    failure_type = np.array(['No Failure'] * num_samples, dtype=object)
    failure_type[twf] = 'TWF'
    failure_type[hdf] = 'HDF'
    failure_type[pwf] = 'PWF'
    failure_type[osf] = 'OSF'
    failure_type[rnf] = 'RNF'
    
    # Handle overlapping failures (prioritize)
    priority_order = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    for i in range(num_samples):
        if any_failure[i]:
            for failure in priority_order:
                if failure_type[i] == failure:
                    break
    
    # Add target columns
    data['Target'] = any_failure.astype(int)
    data['Failure Type'] = failure_type
    
    # Save data if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data.to_csv(save_path, index=False)
        logger.info(f"Sample data saved to {save_path}")
    
    return data

def train_demo_model(data=None, data_path=None, model_path=None):
    """
    Train a simple model for the demo.
    
    Args:
        data: DataFrame with data. If None, data_path must be provided.
        data_path: Path to load data from if data is None.
        model_path: Path to save the trained model.
        
    Returns:
        Trained model pipeline.
    """
    logger.info("Training demo model...")
    
    # Load data if not provided
    if data is None:
        if data_path is None:
            raise ValueError("Either data or data_path must be provided")
        data = pd.read_csv(data_path)
    
    # Load config
    config = load_config()
    
    # Split data
    X = data.drop(['Target', 'Failure Type'], axis=1)
    y = data['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define preprocessing for categorical and numerical features
    categorical_features = ['Type']
    numerical_features = [
        'Air temperature [K]', 
        'Process temperature [K]', 
        'Rotational speed [rpm]', 
        'Torque [Nm]', 
        'Tool wear [min]',
        'Temperature_diff',
        'Power',
        'Tool_wear_rate'
    ]
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ],
        remainder='passthrough'  # This will pass through any columns not specified
    )
    
    # Create feature engineering
    feature_engineering = FeatureEngineering(config)
    
    # Apply feature engineering to training data
    X_train_fe = feature_engineering.transform(X_train)
    
    # Create and train pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ))
    ])
    
    # Fit pipeline
    pipeline.fit(X_train_fe, y_train)
    
    # Apply feature engineering to test data
    X_test_fe = feature_engineering.transform(X_test)
    
    # Evaluate on test set
    accuracy = pipeline.score(X_test_fe, y_test)
    logger.info(f"Model accuracy: {accuracy:.4f}")
    
    # Save model if path provided
    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(pipeline, model_path)
        logger.info(f"Model saved to {model_path}")
    
    return pipeline

def setup_demo():
    """
    Set up the demo by generating data and training a model.
    """
    logger.info("Setting up demo...")
    
    # Load config
    config = load_config()
    
    # Paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
    
    # Ensure directories exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Data paths
    raw_data_path = os.path.join(data_dir, "raw", "maintenance_data.csv")
    
    # Model paths
    model_path = os.path.join(models_dir, "demo_model.pkl")
    
    # Check if data exists
    if not os.path.exists(raw_data_path):
        # Generate sample data
        os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
        data = generate_sample_data(num_samples=10000, save_path=raw_data_path)
    else:
        # Load existing data
        logger.info(f"Using existing data from {raw_data_path}")
        data = pd.read_csv(raw_data_path)
    
    # Train model
    train_demo_model(data=data, model_path=model_path)
    
    logger.info("Demo setup complete!")
    logger.info(f"Data saved to: {raw_data_path}")
    logger.info(f"Model saved to: {model_path}")
    
    # Print next steps
    print("\n" + "="*80)
    print("Demo Setup Complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Run the application: python -m src.deployment.run_app")
    print("2. Open the web UI: http://localhost:8501")
    print("3. Explore the API: http://localhost:8000/docs")
    print("\nEnjoy your predictive maintenance demo!")

if __name__ == "__main__":
    setup_demo()
