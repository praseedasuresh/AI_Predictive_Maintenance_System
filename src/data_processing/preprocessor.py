"""
Data preprocessor for the predictive maintenance model.
This module handles data preprocessing for both training and inference.
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.config_loader import load_config
from src.data_processing.feature_engineering import FeatureEngineering

class DataPreprocessor:
    """
    Class for preprocessing data for the predictive maintenance model.
    """
    
    def __init__(self, config=None):
        """
        Initialize the preprocessor.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config if config is not None else load_config()
        self.feature_engineering = FeatureEngineering(self.config)
        self.preprocessor = None
        self.scaler = None
        self.encoder = None
        self.categorical_features = ['Type']
        self.numerical_features = [
            'Air temperature [K]', 
            'Process temperature [K]', 
            'Rotational speed [rpm]', 
            'Torque [Nm]', 
            'Tool wear [min]'
        ]
        
        # Check if preprocessor exists and load it
        preprocessor_path = self.config['model']['preprocessor_path'] if 'preprocessor_path' in self.config.get('model', {}) else "models/preprocessor.pkl"
        if os.path.exists(preprocessor_path):
            try:
                self.preprocessor = joblib.load(preprocessor_path)
            except Exception as e:
                print(f"Error loading preprocessor: {e}")
    
    def fit(self, X: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit the preprocessor to the data.
        
        Args:
            X: Input features.
            
        Returns:
            Self.
        """
        # Create preprocessor if it doesn't exist
        if self.preprocessor is None:
            # Create transformers for categorical and numerical features
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
            ])
            
            numerical_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            
            # Combine transformers
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', categorical_transformer, self.categorical_features),
                    ('num', numerical_transformer, self.numerical_features)
                ],
                remainder='passthrough'
            )
        
        # Fit the preprocessor
        self.preprocessor.fit(X)
        
        # Save the preprocessor
        preprocessor_path = self.config['model']['preprocessor_path'] if 'preprocessor_path' in self.config.get('model', {}) else "models/preprocessor.pkl"
        os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
        joblib.dump(self.preprocessor, preprocessor_path)
        
        return self
    
    def transform(self, X: pd.DataFrame, apply_feature_engineering: bool = True) -> np.ndarray:
        """
        Transform the data.
        
        Args:
            X: Input features.
            apply_feature_engineering: Whether to apply feature engineering.
            
        Returns:
            Transformed features.
        """
        # Apply feature engineering if requested
        if apply_feature_engineering:
            X = self.feature_engineering.transform(X)
        
        # Check if preprocessor exists
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fitted. Call fit() first.")
        
        # Transform the data
        return self.preprocessor.transform(X)
    
    def fit_transform(self, X: pd.DataFrame, apply_feature_engineering: bool = True) -> np.ndarray:
        """
        Fit and transform the data.
        
        Args:
            X: Input features.
            apply_feature_engineering: Whether to apply feature engineering.
            
        Returns:
            Transformed features.
        """
        # Apply feature engineering if requested
        if apply_feature_engineering:
            X = self.feature_engineering.transform(X)
        
        # Fit and transform
        self.fit(X)
        return self.transform(X, apply_feature_engineering=False)  # Feature engineering already applied
    
    def get_feature_names(self) -> List[str]:
        """
        Get the feature names after transformation.
        
        Returns:
            List of feature names.
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fitted. Call fit() first.")
        
        # Get feature names from column transformer
        cat_features = self.preprocessor.transformers_[0][1].named_steps['onehot'].get_feature_names_out(self.categorical_features)
        
        # Combine with numerical features
        feature_names = list(cat_features) + self.numerical_features
        
        # Add engineered features
        engineered_features = self.feature_engineering.get_feature_names()
        
        return feature_names + engineered_features
    
    def preprocess_for_training(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data for training.
        
        Args:
            data_path: Path to the training data.
            
        Returns:
            Tuple of (X, y) for training.
        """
        # Load data
        data = pd.read_csv(data_path)
        
        # Split into features and target
        X = data.drop(['Target', 'Failure Type'], axis=1, errors='ignore')
        y = data['Target'] if 'Target' in data.columns else None
        
        # Apply preprocessing
        X_processed = self.fit_transform(X)
        
        return X_processed, y
    
    def preprocess_for_inference(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess data for inference.
        
        Args:
            data: Input data.
            
        Returns:
            Preprocessed data.
        """
        # Apply preprocessing
        return self.transform(data)
    
    def save(self, path: str = None) -> None:
        """
        Save the preprocessor.
        
        Args:
            path: Path to save the preprocessor.
        """
        if path is None:
            path = self.config['model']['preprocessor_path'] if 'preprocessor_path' in self.config.get('model', {}) else "models/preprocessor.pkl"
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.preprocessor, path)
    
    def load(self, path: str = None) -> None:
        """
        Load the preprocessor.
        
        Args:
            path: Path to load the preprocessor from.
        """
        if path is None:
            path = self.config['model']['preprocessor_path'] if 'preprocessor_path' in self.config.get('model', {}) else "models/preprocessor.pkl"
        
        if os.path.exists(path):
            self.preprocessor = joblib.load(path)
        else:
            raise FileNotFoundError(f"Preprocessor file not found at {path}")
