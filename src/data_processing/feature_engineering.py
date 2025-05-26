"""
Feature engineering module for the predictive maintenance project.
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from typing import List, Dict, Any, Union, Optional
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.config_loader import load_config

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Class for creating engineered features for predictive maintenance.
    """
    
    def __init__(self, config=None):
        """
        Initialize the feature engineer with configuration.
        
        Args:
            config: Configuration dictionary. If None, uses default settings.
        """
        self.config = config if config is not None else {
            'feature_engineering': {
                'enabled': True,
                'create_polynomial_features': False,
                'create_interaction_features': True,
                'create_power_features': False
            }
        }
        
        # Ensure feature_engineering key exists
        if 'feature_engineering' not in self.config:
            self.config['feature_engineering'] = {
                'enabled': True,
                'create_polynomial_features': False,
                'create_interaction_features': True,
                'create_power_features': False
            }
            
        self.feature_names = None
        
    def fit(self, X, y=None):
        """
        Fit the feature engineer to the data.
        
        Args:
            X: Input features.
            y: Target variable (not used).
            
        Returns:
            Self.
        """
        # Store original feature names
        self.feature_names = X.columns.tolist()
        return self
    
    def transform(self, X):
        """
        Transform the data by creating engineered features.
        
        Args:
            X: Input features.
            
        Returns:
            DataFrame with engineered features.
        """
        # Ensure feature_engineering key exists
        if 'feature_engineering' not in self.config:
            self.config['feature_engineering'] = {
                'enabled': True,
                'create_polynomial_features': False,
                'create_interaction_features': True,
                'create_power_features': False
            }
            
        if not self.config['feature_engineering']['enabled']:
            return X
        
        X_transformed = X.copy()
        
        # Create domain-specific features for predictive maintenance
        X_transformed = self._create_domain_specific_features(X_transformed)
        
        # Create polynomial features if enabled
        if self.config['feature_engineering'].get('create_polynomial_features', False):
            X_transformed = self._create_polynomial_features(X_transformed)
        
        # Create interaction features if enabled
        if self.config['feature_engineering'].get('create_interaction_features', True):
            X_transformed = self._create_interaction_features(X_transformed)
        
        # Create power features if enabled
        if self.config['feature_engineering'].get('create_power_features', False):
            X_transformed = self._create_power_features(X_transformed)
        
        return X_transformed
    
    def _create_domain_specific_features(self, X):
        """
        Create domain-specific features for predictive maintenance.
        
        Args:
            X: Input features.
            
        Returns:
            DataFrame with domain-specific features.
        """
        # Check if required columns exist
        required_columns = [
            'Air temperature [K]', 
            'Process temperature [K]', 
            'Rotational speed [rpm]', 
            'Torque [Nm]', 
            'Tool wear [min]'
        ]
        
        # Only proceed if all required columns exist
        if not all(col in X.columns for col in required_columns):
            return X
        
        # Create a copy to avoid modifying the original
        X_new = X.copy()
        
        # 1. Temperature difference (important for heat dissipation failure)
        X_new['Temperature_diff'] = X_new['Process temperature [K]'] - X_new['Air temperature [K]']
        
        # 2. Power calculation (important for power failure)
        # Convert RPM to rad/s: RPM * 2π/60
        X_new['Power [W]'] = X_new['Rotational speed [rpm]'] * X_new['Torque [Nm]'] * (2 * np.pi / 60)
        
        # 3. Tool wear rate (tool wear per unit of power)
        X_new['Tool_wear_rate'] = X_new['Tool wear [min]'] / (X_new['Power [W]'] + 1)  # Adding 1 to avoid division by zero
        
        # 4. Torque per RPM ratio
        X_new['Torque_per_RPM'] = X_new['Torque [Nm]'] / (X_new['Rotational speed [rpm]'] + 1)  # Adding 1 to avoid division by zero
        
        # 5. Tool wear * Torque (important for overstrain failure)
        X_new['Tool_wear_torque'] = X_new['Tool wear [min]'] * X_new['Torque [Nm]']
        
        # 6. Temperature efficiency ratio
        X_new['Temp_efficiency'] = X_new['Air temperature [K]'] / (X_new['Process temperature [K]'] + 1)  # Adding 1 to avoid division by zero
        
        # 7. Critical operation indicator (combines multiple risk factors)
        X_new['Critical_operation'] = (
            (X_new['Temperature_diff'] < 8.6).astype(int) * 
            (X_new['Power [W]'] < 3500).astype(int) + 
            (X_new['Power [W]'] > 9000).astype(int) +
            (X_new['Tool wear [min]'] > 200).astype(int)
        )
        
        return X_new
    
    def _create_polynomial_features(self, X, degree=2):
        """
        Create polynomial features.
        
        Args:
            X: Input features.
            degree: Degree of polynomial features.
            
        Returns:
            DataFrame with polynomial features.
        """
        # Select only numerical columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if not numerical_cols:
            return X
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(X[numerical_cols])
        
        # Create feature names
        poly_feature_names = poly.get_feature_names_out(numerical_cols)
        
        # Create DataFrame with polynomial features
        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=X.index)
        
        # Drop original features from polynomial DataFrame to avoid duplication
        for col in numerical_cols:
            if col in poly_df.columns:
                poly_df = poly_df.drop(columns=[col])
        
        # Concatenate with original DataFrame
        X_poly = pd.concat([X, poly_df], axis=1)
        
        return X_poly
    
    def _create_interaction_features(self, X):
        """
        Create interaction features between specific columns.
        
        Args:
            X: Input features.
            
        Returns:
            DataFrame with interaction features.
        """
        # Define pairs of features to create interactions for
        interaction_pairs = [
            ('Air temperature [K]', 'Process temperature [K]'),
            ('Rotational speed [rpm]', 'Torque [Nm]'),
            ('Tool wear [min]', 'Torque [Nm]'),
            ('Tool wear [min]', 'Rotational speed [rpm]')
        ]
        
        # Create a copy to avoid modifying the original
        X_new = X.copy()
        
        # Create interaction features
        for col1, col2 in interaction_pairs:
            if col1 in X.columns and col2 in X.columns:
                X_new[f'{col1}_{col2}_interaction'] = X[col1] * X[col2]
        
        return X_new
    
    def _create_power_features(self, X):
        """
        Create power features (square, cube, sqrt) for numerical columns.
        
        Args:
            X: Input features.
            
        Returns:
            DataFrame with power features.
        """
        # Select only numerical columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if not numerical_cols:
            return X
        
        # Create a copy to avoid modifying the original
        X_new = X.copy()
        
        # Create power features
        for col in numerical_cols:
            if col in X.columns:
                X_new[f'{col}_squared'] = X[col] ** 2
                X_new[f'{col}_cubed'] = X[col] ** 3
                X_new[f'{col}_sqrt'] = np.sqrt(np.abs(X[col]))  # Use absolute value to handle negative values
        
        return X_new
    
    def get_feature_names(self):
        """
        Get the names of the engineered features.
        
        Returns:
            List of feature names.
        """
        return [
            'Temperature_diff',
            'Power [W]',
            'Tool_wear_rate',
            'Torque_per_RPM',
            'Tool_wear_torque',
            'Temp_efficiency',
            'Critical_operation'
        ]

# Add the FeatureEngineering class that extends FeatureEngineer to fix the import issue
class FeatureEngineering(FeatureEngineer):
    """
    Class for creating engineered features for predictive maintenance.
    This class extends FeatureEngineer to maintain backward compatibility.
    """
    
    def __init__(self, config=None):
        """
        Initialize the feature engineering with configuration.
        
        Args:
            config: Configuration dictionary. If None, loads the configuration.
        """
        if config is None:
            config = load_config()
            
        # Ensure feature_engineering key exists
        if 'feature_engineering' not in config:
            config['feature_engineering'] = {
                'enabled': True,
                'create_polynomial_features': False,
                'create_interaction_features': True,
                'create_power_features': False
            }
            
        super().__init__(config)

if __name__ == "__main__":
    # Test the feature engineer
    import pandas as pd
    import numpy as np
    
    # Create sample data
    data = pd.DataFrame({
        'Type': ['L', 'M', 'H', 'M', 'L'],
        'Air temperature [K]': [298, 300, 302, 301, 299],
        'Process temperature [K]': [308, 310, 312, 311, 309],
        'Rotational speed [rpm]': [1500, 1550, 1600, 1550, 1500],
        'Torque [Nm]': [35, 40, 45, 38, 37],
        'Tool wear [min]': [100, 120, 150, 130, 110]
    })
    
    # Create feature engineer
    fe = FeatureEngineering()
    
    # Transform data
    data_transformed = fe.transform(data)
    
    # Print results
    print("Original data shape:", data.shape)
    print("Transformed data shape:", data_transformed.shape)
    print("New features:", [col for col in data_transformed.columns if col not in data.columns])
