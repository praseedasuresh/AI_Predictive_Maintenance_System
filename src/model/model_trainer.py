"""
Model training module for the predictive maintenance project.
"""
import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.config_loader import load_config, get_model_params
from src.data_processing.feature_engineering import FeatureEngineer

class ModelTrainer:
    """
    Class for training and evaluating machine learning models for predictive maintenance.
    """
    
    def __init__(self, config=None):
        """
        Initialize the model trainer with configuration.
        
        Args:
            config: Configuration dictionary. If None, loads the configuration.
        """
        self.config = config if config is not None else load_config()
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = -float('inf')
        self.results = {}
        self.feature_engineer = FeatureEngineer(self.config)
        
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        """
        Train a Random Forest model.
        
        Args:
            X_train: Training features.
            y_train: Training target.
            
        Returns:
            Trained Random Forest model.
        """
        model_params = get_model_params('random_forest', self.config)
        if not model_params['enabled']:
            return None
        
        # Extract parameters
        params = {
            'n_estimators': model_params['n_estimators'],
            'max_depth': model_params['max_depth'],
            'min_samples_split': model_params['min_samples_split'],
            'min_samples_leaf': model_params['min_samples_leaf'],
            'class_weight': model_params['class_weight'],
            'random_state': self.config['data']['random_state']
        }
        
        # Create pipeline with feature engineering
        pipeline = Pipeline([
            ('feature_engineering', self.feature_engineer),
            ('classifier', RandomForestClassifier(**params))
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        return pipeline
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
        """
        Train an XGBoost model.
        
        Args:
            X_train: Training features.
            y_train: Training target.
            
        Returns:
            Trained XGBoost model.
        """
        model_params = get_model_params('xgboost', self.config)
        if not model_params['enabled']:
            return None
        
        # Extract parameters
        params = {
            'n_estimators': model_params['n_estimators'],
            'learning_rate': model_params['learning_rate'],
            'max_depth': model_params['max_depth'],
            'subsample': model_params['subsample'],
            'colsample_bytree': model_params['colsample_bytree'],
            'scale_pos_weight': model_params['scale_pos_weight'],
            'random_state': self.config['data']['random_state']
        }
        
        # Create pipeline with feature engineering
        pipeline = Pipeline([
            ('feature_engineering', self.feature_engineer),
            ('classifier', xgb.XGBClassifier(**params))
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        return pipeline
    
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
        """
        Train a Logistic Regression model.
        
        Args:
            X_train: Training features.
            y_train: Training target.
            
        Returns:
            Trained Logistic Regression model.
        """
        model_params = get_model_params('logistic_regression', self.config)
        if not model_params['enabled']:
            return None
        
        # Extract parameters
        params = {
            'C': model_params['C'],
            'penalty': model_params['penalty'],
            'solver': model_params['solver'],
            'class_weight': model_params['class_weight'],
            'random_state': self.config['data']['random_state']
        }
        
        # Create pipeline with feature engineering
        pipeline = Pipeline([
            ('feature_engineering', self.feature_engineer),
            ('classifier', LogisticRegression(**params))
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        return pipeline
    
    def train_svm(self, X_train: pd.DataFrame, y_train: pd.Series) -> SVC:
        """
        Train a Support Vector Machine model.
        
        Args:
            X_train: Training features.
            y_train: Training target.
            
        Returns:
            Trained SVM model.
        """
        model_params = get_model_params('svm', self.config)
        if not model_params['enabled']:
            return None
        
        # Extract parameters
        params = {
            'C': model_params['C'],
            'kernel': model_params['kernel'],
            'gamma': model_params['gamma'],
            'probability': model_params['probability'],
            'class_weight': model_params['class_weight'],
            'random_state': self.config['data']['random_state']
        }
        
        # Create pipeline with feature engineering
        pipeline = Pipeline([
            ('feature_engineering', self.feature_engineer),
            ('classifier', SVC(**params))
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        return pipeline
    
    def train_neural_network(self, X_train: pd.DataFrame, y_train: pd.Series, 
                            X_val: pd.DataFrame = None, y_val: pd.Series = None) -> keras.Model:
        """
        Train a Neural Network model.
        
        Args:
            X_train: Training features.
            y_train: Training target.
            X_val: Validation features.
            y_val: Validation target.
            
        Returns:
            Trained Neural Network model.
        """
        model_params = get_model_params('neural_network', self.config)
        if not model_params['enabled']:
            return None
        
        # Apply feature engineering
        X_train_fe = self.feature_engineer.fit_transform(X_train)
        if X_val is not None:
            X_val_fe = self.feature_engineer.transform(X_val)
        
        # Extract parameters
        architecture = model_params['architecture']
        activation = model_params['activation']
        dropout_rate = model_params['dropout_rate']
        batch_size = model_params['batch_size']
        epochs = model_params['epochs']
        early_stopping = model_params['early_stopping']
        patience = model_params['patience']
        
        # Determine the number of input features
        input_dim = X_train_fe.shape[1]
        
        # Determine if this is a binary or multiclass problem
        num_classes = len(np.unique(y_train))
        is_binary = num_classes == 2
        
        # Define the output layer and loss function
        if is_binary:
            output_units = 1
            output_activation = 'sigmoid'
            loss = 'binary_crossentropy'
        else:
            output_units = num_classes
            output_activation = 'softmax'
            loss = 'sparse_categorical_crossentropy'
        
        # Build the model
        model = keras.Sequential()
        
        # Input layer
        model.add(keras.layers.Input(shape=(input_dim,)))
        
        # Hidden layers
        for units in architecture:
            model.add(keras.layers.Dense(units, activation=activation))
            model.add(keras.layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(keras.layers.Dense(output_units, activation=output_activation))
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss=loss,
            metrics=['accuracy']
        )
        
        # Define callbacks
        callbacks = []
        if early_stopping and X_val is not None and y_val is not None:
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
            ))
        
        # Train the model
        if X_val is not None and y_val is not None:
            history = model.fit(
                X_train_fe, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val_fe, y_val),
                callbacks=callbacks,
                verbose=1
            )
        else:
            history = model.fit(
                X_train_fe, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
        
        # Create a wrapper class to handle feature engineering and prediction
        class NeuralNetworkWrapper:
            def __init__(self, model, feature_engineer):
                self.model = model
                self.feature_engineer = feature_engineer
                
            def predict(self, X):
                X_fe = self.feature_engineer.transform(X)
                if is_binary:
                    y_pred_proba = self.model.predict(X_fe)
                    threshold = 0.5  # Default threshold
                    y_pred = (y_pred_proba > threshold).astype(int)
                    return y_pred.flatten()
                else:
                    return np.argmax(self.model.predict(X_fe), axis=1)
                
            def predict_proba(self, X):
                X_fe = self.feature_engineer.transform(X)
                return self.model.predict(X_fe)
        
        return NeuralNetworkWrapper(model, self.feature_engineer)
    
    def perform_hyperparameter_tuning(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for a specific model.
        
        Args:
            model_name: Name of the model to tune.
            X_train: Training features.
            y_train: Training target.
            
        Returns:
            Dictionary of best hyperparameters.
        """
        tuning_config = self.config['training']['hyperparameter_tuning']
        if not tuning_config['enabled']:
            return None
        
        method = tuning_config['method']
        n_iter = tuning_config['n_iter']
        cv = tuning_config['cv']
        
        # Define parameter grids for each model
        param_grids = {
            'random_forest': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [None, 10, 20, 30],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4]
            },
            'xgboost': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__max_depth': [3, 5, 7],
                'classifier__subsample': [0.6, 0.8, 1.0],
                'classifier__colsample_bytree': [0.6, 0.8, 1.0],
                'classifier__scale_pos_weight': [1.0, 3.0, 5.0]
            },
            'logistic_regression': {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear', 'saga']
            },
            'svm': {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__kernel': ['linear', 'rbf', 'poly'],
                'classifier__gamma': ['scale', 'auto', 0.1, 1.0]
            }
        }
        
        if model_name not in param_grids:
            return None
        
        # Get the base pipeline and parameter grid
        if model_name == 'random_forest':
            base_pipeline = Pipeline([
                ('feature_engineering', self.feature_engineer),
                ('classifier', RandomForestClassifier(random_state=self.config['data']['random_state']))
            ])
        elif model_name == 'xgboost':
            base_pipeline = Pipeline([
                ('feature_engineering', self.feature_engineer),
                ('classifier', xgb.XGBClassifier(random_state=self.config['data']['random_state']))
            ])
        elif model_name == 'logistic_regression':
            base_pipeline = Pipeline([
                ('feature_engineering', self.feature_engineer),
                ('classifier', LogisticRegression(random_state=self.config['data']['random_state']))
            ])
        elif model_name == 'svm':
            base_pipeline = Pipeline([
                ('feature_engineering', self.feature_engineer),
                ('classifier', SVC(random_state=self.config['data']['random_state'], probability=True))
            ])
        else:
            return None
        
        param_grid = param_grids[model_name]
        
        # Perform hyperparameter tuning
        if method == 'grid':
            search = GridSearchCV(
                base_pipeline,
                param_grid,
                cv=cv,
                scoring='f1',  # Using F1 score for imbalanced classification
                n_jobs=-1
            )
        elif method == 'random':
            search = RandomizedSearchCV(
                base_pipeline,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring='f1',  # Using F1 score for imbalanced classification
                random_state=self.config['data']['random_state'],
                n_jobs=-1
            )
        else:
            return None
        
        # Fit the search
        search.fit(X_train, y_train)
        
        # Extract the best parameters for the classifier
        best_params = {}
        for param, value in search.best_params_.items():
            if param.startswith('classifier__'):
                best_params[param.replace('classifier__', '')] = value
        
        return best_params
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, Any]:
        """
        Train all enabled models.
        
        Args:
            X_train: Training features.
            y_train: Training target.
            X_val: Validation features.
            y_val: Validation target.
            
        Returns:
            Dictionary of trained models.
        """
        # Train Random Forest
        if self.config['models']['random_forest']['enabled']:
            print("Training Random Forest model...")
            if self.config['training']['hyperparameter_tuning']['enabled']:
                best_params = self.perform_hyperparameter_tuning('random_forest', X_train, y_train)
                if best_params:
                    # Update parameters with best values
                    for param, value in best_params.items():
                        self.config['models']['random_forest'][param] = value
            
            rf_model = self.train_random_forest(X_train, y_train)
            self.models['random_forest'] = rf_model
            
            # Evaluate on validation set if available
            if X_val is not None and y_val is not None:
                rf_score = self.evaluate_model(rf_model, X_val, y_val)
                self.results['random_forest'] = rf_score
                
                # Update best model if this is better
                if rf_score['f1'] > self.best_score:  # Using F1 score for imbalanced data
                    self.best_model = rf_model
                    self.best_model_name = 'random_forest'
                    self.best_score = rf_score['f1']
        
        # Train XGBoost
        if self.config['models']['xgboost']['enabled']:
            print("Training XGBoost model...")
            if self.config['training']['hyperparameter_tuning']['enabled']:
                best_params = self.perform_hyperparameter_tuning('xgboost', X_train, y_train)
                if best_params:
                    # Update parameters with best values
                    for param, value in best_params.items():
                        self.config['models']['xgboost'][param] = value
            
            xgb_model = self.train_xgboost(X_train, y_train)
            self.models['xgboost'] = xgb_model
            
            # Evaluate on validation set if available
            if X_val is not None and y_val is not None:
                xgb_score = self.evaluate_model(xgb_model, X_val, y_val)
                self.results['xgboost'] = xgb_score
                
                # Update best model if this is better
                if xgb_score['f1'] > self.best_score:
                    self.best_model = xgb_model
                    self.best_model_name = 'xgboost'
                    self.best_score = xgb_score['f1']
        
        # Train Logistic Regression
        if self.config['models']['logistic_regression']['enabled']:
            print("Training Logistic Regression model...")
            if self.config['training']['hyperparameter_tuning']['enabled']:
                best_params = self.perform_hyperparameter_tuning('logistic_regression', X_train, y_train)
                if best_params:
                    # Update parameters with best values
                    for param, value in best_params.items():
                        self.config['models']['logistic_regression'][param] = value
            
            lr_model = self.train_logistic_regression(X_train, y_train)
            self.models['logistic_regression'] = lr_model
            
            # Evaluate on validation set if available
            if X_val is not None and y_val is not None:
                lr_score = self.evaluate_model(lr_model, X_val, y_val)
                self.results['logistic_regression'] = lr_score
                
                # Update best model if this is better
                if lr_score['f1'] > self.best_score:
                    self.best_model = lr_model
                    self.best_model_name = 'logistic_regression'
                    self.best_score = lr_score['f1']
        
        # Train SVM
        if self.config['models']['svm']['enabled']:
            print("Training SVM model...")
            if self.config['training']['hyperparameter_tuning']['enabled']:
                best_params = self.perform_hyperparameter_tuning('svm', X_train, y_train)
                if best_params:
                    # Update parameters with best values
                    for param, value in best_params.items():
                        self.config['models']['svm'][param] = value
            
            svm_model = self.train_svm(X_train, y_train)
            self.models['svm'] = svm_model
            
            # Evaluate on validation set if available
            if X_val is not None and y_val is not None:
                svm_score = self.evaluate_model(svm_model, X_val, y_val)
                self.results['svm'] = svm_score
                
                # Update best model if this is better
                if svm_score['f1'] > self.best_score:
                    self.best_model = svm_model
                    self.best_model_name = 'svm'
                    self.best_score = svm_score['f1']
        
        # Train Neural Network
        if self.config['models']['neural_network']['enabled']:
            print("Training Neural Network model...")
            nn_model = self.train_neural_network(X_train, y_train, X_val, y_val)
            self.models['neural_network'] = nn_model
            
            # Evaluate on validation set if available
            if X_val is not None and y_val is not None:
                nn_score = self.evaluate_model(nn_model, X_val, y_val)
                self.results['neural_network'] = nn_score
                
                # Update best model if this is better
                if nn_score['f1'] > self.best_score:
                    self.best_model = nn_model
                    self.best_model_name = 'neural_network'
                    self.best_score = nn_score['f1']
        
        return self.models
    
    def evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate a model on the given data.
        
        Args:
            model: Trained model.
            X: Features.
            y: Target.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        # Get the metrics to calculate
        metrics = self.config['evaluation']['metrics']
        results = {}
        
        # Make predictions
        if hasattr(model, 'predict'):
            y_pred = model.predict(X)
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X)
            else:
                y_pred_proba = None
        else:
            # For pipeline models
            y_pred = model.predict(X)
            if hasattr(model.named_steps['classifier'], 'predict_proba'):
                y_pred_proba = model.predict_proba(X)
            else:
                y_pred_proba = None
        
        # Calculate metrics
        if 'accuracy' in metrics:
            results['accuracy'] = accuracy_score(y, y_pred)
        
        if 'precision' in metrics:
            results['precision'] = precision_score(y, y_pred, average='weighted')
        
        if 'recall' in metrics:
            results['recall'] = recall_score(y, y_pred, average='weighted')
        
        if 'f1' in metrics:
            results['f1'] = f1_score(y, y_pred, average='weighted')
        
        if 'roc_auc' in metrics and y_pred_proba is not None:
            # ROC AUC is only applicable for binary classification
            if len(np.unique(y)) == 2:
                if y_pred_proba.shape[1] == 2:
                    results['roc_auc'] = roc_auc_score(y, y_pred_proba[:, 1])
        
        return results
    
    def save_model(self, model: Any, model_name: str, output_dir: str = None) -> str:
        """
        Save a trained model to disk.
        
        Args:
            model: Trained model to save.
            model_name: Name of the model.
            output_dir: Directory to save the model. If None, uses the value from config.
            
        Returns:
            Path to the saved model.
        """
        if output_dir is None:
            # Get the project root directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
            output_dir = os.path.join(project_root, "models")
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the model
        if isinstance(model, Pipeline) or hasattr(model, 'predict'):
            # Save scikit-learn model or custom wrapper
            model_path = os.path.join(output_dir, f"{model_name}_model.pkl")
            joblib.dump(model, model_path)
        else:
            # Fallback for other model types
            model_path = os.path.join(output_dir, f"{model_name}_model.pkl")
            joblib.dump(model, model_path)
        
        return model_path
    
    def load_model(self, model_path: str) -> Any:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model.
            
        Returns:
            Loaded model.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load the model
        model = joblib.load(model_path)
        
        return model
    
    def get_feature_importance(self, model: Any) -> pd.DataFrame:
        """
        Get feature importance from the model if available.
        
        Args:
            model: Trained model.
            
        Returns:
            DataFrame with feature importance.
        """
        # Check if model is a pipeline
        if isinstance(model, Pipeline):
            if hasattr(model.named_steps['classifier'], 'feature_importances_'):
                # Get feature names after feature engineering
                X_dummy = pd.DataFrame(np.zeros((1, 5)), columns=[
                    'Air temperature [K]', 'Process temperature [K]', 
                    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
                ])
                X_transformed = model.named_steps['feature_engineering'].transform(X_dummy)
                feature_names = X_transformed.columns.tolist()
                
                # Get feature importances
                importances = model.named_steps['classifier'].feature_importances_
                
                # Create DataFrame
                importance_df = pd.DataFrame({
                    'Feature': feature_names[:len(importances)],
                    'Importance': importances
                })
                
                # Sort by importance
                importance_df = importance_df.sort_values('Importance', ascending=False)
                
                return importance_df
        
        # For models with direct feature_importances_ attribute
        elif hasattr(model, 'feature_importances_'):
            # For XGBoost models
            if isinstance(model, xgb.XGBClassifier):
                feature_names = model.get_booster().feature_names
                importances = model.feature_importances_
            else:
                # For other models
                feature_names = [f"Feature_{i}" for i in range(len(model.feature_importances_))]
                importances = model.feature_importances_
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            return importance_df
        
        # For models with coef_ attribute (like logistic regression)
        elif hasattr(model, 'coef_'):
            # Get feature names
            feature_names = [f"Feature_{i}" for i in range(model.coef_.shape[1])]
            
            # Get coefficients
            if len(model.coef_.shape) > 1:
                importances = np.abs(model.coef_[0])
            else:
                importances = np.abs(model.coef_)
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            return importance_df
        
        return None

if __name__ == "__main__":
    # Test the model trainer
    config = load_config()
    trainer = ModelTrainer(config)
    
    # Example usage
    print("Model Trainer initialized successfully.")
    print(f"Configuration: {config['models'].keys()}")
