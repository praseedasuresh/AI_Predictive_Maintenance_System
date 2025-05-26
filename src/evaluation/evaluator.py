"""
Model evaluation module for the AI model development project.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
import joblib
from tensorflow import keras

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.config_loader import load_config

class ModelEvaluator:
    """
    Class for evaluating and analyzing model performance.
    """
    
    def __init__(self, config=None):
        """
        Initialize the evaluator with configuration.
        
        Args:
            config: Configuration dictionary. If None, loads the configuration.
        """
        self.config = config if config is not None else load_config()
        self.results = {}
        
    def evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series, model_name: str) -> Dict[str, float]:
        """
        Evaluate a model on the given data.
        
        Args:
            model: Trained model.
            X: Features.
            y: Target.
            model_name: Name of the model.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        # Get the metrics to calculate
        metrics = self.config['evaluation']['metrics']
        results = {}
        
        # Make predictions
        if isinstance(model, keras.Model):
            # For neural networks
            y_pred_proba = model.predict(X)
            if y_pred_proba.shape[1] > 1:  # Multiclass
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:  # Binary
                threshold = self.config['evaluation']['threshold']
                y_pred = (y_pred_proba > threshold).astype(int)
                y_pred = y_pred.flatten()
        else:
            # For sklearn models
            y_pred = model.predict(X)
            if hasattr(model, 'predict_proba'):
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
                if isinstance(model, keras.Model) and y_pred_proba.shape[1] == 1:
                    results['roc_auc'] = roc_auc_score(y, y_pred_proba)
                elif y_pred_proba.shape[1] == 2:
                    results['roc_auc'] = roc_auc_score(y, y_pred_proba[:, 1])
        
        # Store the results
        self.results[model_name] = results
        
        return results
    
    def generate_classification_report(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                                      output_path: str = None) -> str:
        """
        Generate a detailed classification report.
        
        Args:
            model: Trained model.
            X: Features.
            y: Target.
            output_path: Path to save the report. If None, returns the report as a string.
            
        Returns:
            Classification report as a string.
        """
        # Make predictions
        if isinstance(model, keras.Model):
            # For neural networks
            y_pred_proba = model.predict(X)
            if y_pred_proba.shape[1] > 1:  # Multiclass
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:  # Binary
                threshold = self.config['evaluation']['threshold']
                y_pred = (y_pred_proba > threshold).astype(int)
                y_pred = y_pred.flatten()
        else:
            # For sklearn models
            y_pred = model.predict(X)
        
        # Generate classification report
        report = classification_report(y, y_pred, output_dict=False)
        
        # Save the report if output_path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
        
        return report
    
    def plot_confusion_matrix(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                             output_path: str = None) -> None:
        """
        Plot confusion matrix for a model.
        
        Args:
            model: Trained model.
            X: Features.
            y: Target.
            output_path: Path to save the plot. If None, displays the plot.
        """
        # Make predictions
        if isinstance(model, keras.Model):
            y_pred_proba = model.predict(X)
            if y_pred_proba.shape[1] > 1:  # Multiclass
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:  # Binary
                threshold = self.config['evaluation']['threshold']
                y_pred = (y_pred_proba > threshold).astype(int)
                y_pred = y_pred.flatten()
        else:
            y_pred = model.predict(X)
        
        # Compute confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Save or display the plot
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def plot_roc_curve(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                      output_path: str = None) -> None:
        """
        Plot ROC curve for a binary classification model.
        
        Args:
            model: Trained model.
            X: Features.
            y: Target.
            output_path: Path to save the plot. If None, displays the plot.
        """
        # Check if binary classification
        if len(np.unique(y)) != 2:
            print("ROC curve is only applicable for binary classification.")
            return
        
        # Get predicted probabilities
        if isinstance(model, keras.Model):
            y_pred_proba = model.predict(X).flatten()
        else:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X)[:, 1]
            else:
                print("Model does not support probability predictions.")
                return
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
        
        # Compute AUC
        auc = roc_auc_score(y, y_pred_proba)
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # Save or display the plot
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def plot_precision_recall_curve(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                                  output_path: str = None) -> None:
        """
        Plot precision-recall curve for a binary classification model.
        
        Args:
            model: Trained model.
            X: Features.
            y: Target.
            output_path: Path to save the plot. If None, displays the plot.
        """
        # Check if binary classification
        if len(np.unique(y)) != 2:
            print("Precision-Recall curve is only applicable for binary classification.")
            return
        
        # Get predicted probabilities
        if isinstance(model, keras.Model):
            y_pred_proba = model.predict(X).flatten()
        else:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X)[:, 1]
            else:
                print("Model does not support probability predictions.")
                return
        
        # Compute precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y, y_pred_proba)
        
        # Compute average precision
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(y, y_pred_proba)
        
        # Plot precision-recall curve
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, label=f'Precision-Recall curve (AP = {ap:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        # Save or display the plot
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def plot_feature_importance(self, model: Any, feature_names: List[str], 
                               output_path: str = None) -> None:
        """
        Plot feature importance for tree-based models.
        
        Args:
            model: Trained model.
            feature_names: List of feature names.
            output_path: Path to save the plot. If None, displays the plot.
        """
        if not hasattr(model, 'feature_importances_'):
            print("Model does not support feature importance visualization.")
            return
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        
        # Rearrange feature names so they match the sorted feature importances
        names = [feature_names[i] for i in indices]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importance")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), names, rotation=90)
        plt.tight_layout()
        
        # Save or display the plot
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def generate_evaluation_report(self, output_path: str = None) -> Dict[str, Any]:
        """
        Generate an evaluation report for all evaluated models.
        
        Args:
            output_path: Path to save the report. If None, returns the report as a dictionary.
            
        Returns:
            Dictionary containing the evaluation report.
        """
        # Find the best model
        best_model_name = None
        best_score = -float('inf')
        
        for model_name, results in self.results.items():
            if 'accuracy' in results and results['accuracy'] > best_score:
                best_score = results['accuracy']
                best_model_name = model_name
        
        # Create the report
        report = {
            'models': self.results,
            'best_model': {
                'name': best_model_name,
                'score': best_score
            }
        }
        
        # Save the report if output_path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=4)
        
        return report
    
    def analyze_errors(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                      feature_names: List[str] = None) -> pd.DataFrame:
        """
        Analyze prediction errors to identify patterns.
        
        Args:
            model: Trained model.
            X: Features.
            y: Target.
            feature_names: List of feature names. If None, uses X.columns.
            
        Returns:
            DataFrame containing error analysis.
        """
        # Use column names if feature_names is not provided
        if feature_names is None and hasattr(X, 'columns'):
            feature_names = X.columns.tolist()
        
        # Make predictions
        if isinstance(model, keras.Model):
            # For neural networks
            y_pred_proba = model.predict(X)
            if y_pred_proba.shape[1] > 1:  # Multiclass
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:  # Binary
                threshold = self.config['evaluation']['threshold']
                y_pred = (y_pred_proba > threshold).astype(int)
                y_pred = y_pred.flatten()
        else:
            # For sklearn models
            y_pred = model.predict(X)
        
        # Create a DataFrame with features and predictions
        if hasattr(X, 'values'):
            data = X.values
        else:
            data = X
            
        error_df = pd.DataFrame(data, columns=feature_names)
        error_df['true_label'] = y
        error_df['predicted_label'] = y_pred
        error_df['is_error'] = (y != y_pred)
        
        # Add probability information if available
        if isinstance(model, keras.Model) or hasattr(model, 'predict_proba'):
            if isinstance(model, keras.Model):
                if y_pred_proba.shape[1] > 1:  # Multiclass
                    for i in range(y_pred_proba.shape[1]):
                        error_df[f'prob_class_{i}'] = y_pred_proba[:, i]
                else:  # Binary
                    error_df['probability'] = y_pred_proba.flatten()
            else:
                y_pred_proba = model.predict_proba(X)
                for i in range(y_pred_proba.shape[1]):
                    error_df[f'prob_class_{i}'] = y_pred_proba[:, i]
        
        return error_df
    
    def plot_error_distribution(self, error_df: pd.DataFrame, feature: str, 
                               output_path: str = None) -> None:
        """
        Plot the distribution of errors for a specific feature.
        
        Args:
            error_df: DataFrame containing error analysis.
            feature: Feature to analyze.
            output_path: Path to save the plot. If None, displays the plot.
        """
        plt.figure(figsize=(12, 6))
        
        # Check if the feature is categorical or numerical
        if error_df[feature].dtype == 'object' or error_df[feature].nunique() < 10:
            # Categorical feature
            sns.countplot(x=feature, hue='is_error', data=error_df)
            plt.xticks(rotation=45)
        else:
            # Numerical feature
            plt.figure(figsize=(12, 6))
            sns.histplot(data=error_df, x=feature, hue='is_error', multiple='stack', bins=20)
        
        plt.title(f'Error Distribution for {feature}')
        plt.tight_layout()
        
        # Save or display the plot
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def generate_error_analysis_report(self, error_df: pd.DataFrame, 
                                     output_path: str = None) -> Dict[str, Any]:
        """
        Generate a comprehensive error analysis report.
        
        Args:
            error_df: DataFrame containing error analysis.
            output_path: Path to save the report. If None, returns the report as a dictionary.
            
        Returns:
            Dictionary containing the error analysis report.
        """
        # Calculate overall error rate
        error_rate = error_df['is_error'].mean()
        
        # Calculate error rate by class
        error_by_class = error_df.groupby('true_label')['is_error'].mean().to_dict()
        
        # Find the most common misclassifications
        misclassifications = error_df[error_df['is_error']].groupby(
            ['true_label', 'predicted_label']).size().reset_index(name='count')
        misclassifications = misclassifications.sort_values('count', ascending=False)
        
        # Create the report
        report = {
            'overall_error_rate': error_rate,
            'error_by_class': error_by_class,
            'top_misclassifications': misclassifications.head(5).to_dict('records')
        }
        
        # Save the report if output_path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=4)
        
        return report

if __name__ == "__main__":
    # Test the evaluator
    config = load_config()
    evaluator = ModelEvaluator(config)
    
    # Example usage
    print("Model Evaluator initialized successfully.")
    print(f"Evaluation metrics: {config['evaluation']['metrics']}")
