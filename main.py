"""
Main script for the AI model development project.
This script orchestrates the entire workflow from data processing to model training and evaluation.
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

from src.utils.config_loader import load_config
from src.data_processing.preprocessor import DataPreprocessor
from src.model.model_trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AI Model Development Pipeline')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to data file (overrides config)')
    parser.add_argument('--output', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'predict'], default='train',
                        help='Mode to run the pipeline in')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to saved model (for evaluate or predict modes)')
    
    return parser.parse_args()

def setup_output_directory(output_dir):
    """Set up the output directory for results."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
    
    return output_dir

def train_pipeline(config, data_path=None, output_dir='results'):
    """
    Run the training pipeline.
    
    Args:
        config: Configuration dictionary.
        data_path: Path to data file. If None, uses the path from config.
        output_dir: Directory to save results.
    """
    print("Starting training pipeline...")
    
    # Set up output directory
    output_dir = setup_output_directory(output_dir)
    
    # Initialize components
    preprocessor = DataPreprocessor(config)
    trainer = ModelTrainer(config)
    evaluator = ModelEvaluator(config)
    
    # Get data path from config if not provided
    if data_path is None:
        # Get the project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = current_dir
        data_path = os.path.join(project_root, config['data']['train_path'])
    
    # Prepare data
    print(f"Preparing data from {data_path}...")
    try:
        X_train, y_train, X_val, y_val = preprocessor.prepare_train_validation_data(data_path)
        print(f"Data prepared. Training set: {X_train.shape}, Validation set: {X_val.shape}")
    except Exception as e:
        print(f"Error preparing data: {e}")
        print("Using synthetic data for demonstration...")
        # Create synthetic data for demonstration
        X_train = pd.DataFrame(np.random.randn(100, 10), columns=[f'feature_{i}' for i in range(10)])
        y_train = pd.Series(np.random.randint(0, 2, 100))
        X_val = pd.DataFrame(np.random.randn(30, 10), columns=[f'feature_{i}' for i in range(10)])
        y_val = pd.Series(np.random.randint(0, 2, 30))
    
    # Train models
    print("Training models...")
    models = trainer.train_models(X_train, y_train, X_val, y_val)
    
    # Save the best model
    if trainer.best_model is not None:
        best_model_path = trainer.save_model(
            trainer.best_model, 
            trainer.best_model_name,
            os.path.join(output_dir, 'models')
        )
        print(f"Best model ({trainer.best_model_name}) saved to {best_model_path}")
    
    # Evaluate models
    print("Evaluating models...")
    for model_name, model in models.items():
        if model is not None:
            # Evaluate on validation set
            results = evaluator.evaluate_model(model, X_val, y_val, model_name)
            print(f"Model: {model_name}, Accuracy: {results.get('accuracy', 'N/A'):.4f}")
            
            # Generate classification report
            report = evaluator.generate_classification_report(
                model, X_val, y_val,
                os.path.join(output_dir, 'reports', f'{model_name}_classification_report.txt')
            )
            
            # Plot confusion matrix
            evaluator.plot_confusion_matrix(
                model, X_val, y_val,
                os.path.join(output_dir, 'plots', f'{model_name}_confusion_matrix.png')
            )
            
            # Plot ROC curve for binary classification
            if len(np.unique(y_val)) == 2:
                evaluator.plot_roc_curve(
                    model, X_val, y_val,
                    os.path.join(output_dir, 'plots', f'{model_name}_roc_curve.png')
                )
                
                evaluator.plot_precision_recall_curve(
                    model, X_val, y_val,
                    os.path.join(output_dir, 'plots', f'{model_name}_precision_recall_curve.png')
                )
            
            # Plot feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                evaluator.plot_feature_importance(
                    model, X_train.columns.tolist(),
                    os.path.join(output_dir, 'plots', f'{model_name}_feature_importance.png')
                )
    
    # Generate evaluation report
    report = evaluator.generate_evaluation_report(
        os.path.join(output_dir, 'reports', 'evaluation_report.json')
    )
    
    # Analyze errors
    if trainer.best_model is not None:
        error_df = evaluator.analyze_errors(trainer.best_model, X_val, y_val)
        
        # Generate error analysis report
        error_report = evaluator.generate_error_analysis_report(
            error_df,
            os.path.join(output_dir, 'reports', 'error_analysis_report.json')
        )
        
        # Plot error distributions for a few features
        for feature in X_train.columns[:3]:  # Plot first 3 features
            evaluator.plot_error_distribution(
                error_df, feature,
                os.path.join(output_dir, 'plots', f'error_distribution_{feature}.png')
            )
    
    print(f"Training pipeline completed. Results saved to {output_dir}")
    return output_dir

def evaluate_pipeline(config, model_path, data_path=None, output_dir='results'):
    """
    Run the evaluation pipeline on a saved model.
    
    Args:
        config: Configuration dictionary.
        model_path: Path to saved model.
        data_path: Path to data file. If None, uses the path from config.
        output_dir: Directory to save results.
    """
    print("Starting evaluation pipeline...")
    
    # Set up output directory
    output_dir = setup_output_directory(output_dir)
    
    # Initialize components
    preprocessor = DataPreprocessor(config)
    trainer = ModelTrainer(config)
    evaluator = ModelEvaluator(config)
    
    # Load the model
    try:
        model = trainer.load_model(model_path)
        model_name = os.path.basename(model_path).split('_')[0]
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Get data path from config if not provided
    if data_path is None:
        # Get the project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = current_dir
        data_path = os.path.join(project_root, config['data']['test_path'])
    
    # Prepare data
    print(f"Preparing data from {data_path}...")
    try:
        X_test, y_test = preprocessor.prepare_test_data(data_path)
        print(f"Data prepared. Test set: {X_test.shape}")
    except Exception as e:
        print(f"Error preparing data: {e}")
        print("Using synthetic data for demonstration...")
        # Create synthetic data for demonstration
        X_test = pd.DataFrame(np.random.randn(50, 10), columns=[f'feature_{i}' for i in range(10)])
        y_test = pd.Series(np.random.randint(0, 2, 50))
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluator.evaluate_model(model, X_test, y_test, model_name)
    print(f"Model: {model_name}, Accuracy: {results.get('accuracy', 'N/A'):.4f}")
    
    # Generate classification report
    report = evaluator.generate_classification_report(
        model, X_test, y_test,
        os.path.join(output_dir, 'reports', f'{model_name}_classification_report.txt')
    )
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(
        model, X_test, y_test,
        os.path.join(output_dir, 'plots', f'{model_name}_confusion_matrix.png')
    )
    
    # Plot ROC curve for binary classification
    if len(np.unique(y_test)) == 2:
        evaluator.plot_roc_curve(
            model, X_test, y_test,
            os.path.join(output_dir, 'plots', f'{model_name}_roc_curve.png')
        )
        
        evaluator.plot_precision_recall_curve(
            model, X_test, y_test,
            os.path.join(output_dir, 'plots', f'{model_name}_precision_recall_curve.png')
        )
    
    # Plot feature importance for tree-based models
    if hasattr(model, 'feature_importances_'):
        evaluator.plot_feature_importance(
            model, X_test.columns.tolist(),
            os.path.join(output_dir, 'plots', f'{model_name}_feature_importance.png')
        )
    
    # Generate evaluation report
    report = evaluator.generate_evaluation_report(
        os.path.join(output_dir, 'reports', 'evaluation_report.json')
    )
    
    # Analyze errors
    error_df = evaluator.analyze_errors(model, X_test, y_test)
    
    # Generate error analysis report
    error_report = evaluator.generate_error_analysis_report(
        error_df,
        os.path.join(output_dir, 'reports', 'error_analysis_report.json')
    )
    
    # Plot error distributions for a few features
    for feature in X_test.columns[:3]:  # Plot first 3 features
        evaluator.plot_error_distribution(
            error_df, feature,
            os.path.join(output_dir, 'plots', f'error_distribution_{feature}.png')
        )
    
    print(f"Evaluation pipeline completed. Results saved to {output_dir}")
    return output_dir

def predict_pipeline(config, model_path, data_path=None, output_dir='results'):
    """
    Run the prediction pipeline on a saved model.
    
    Args:
        config: Configuration dictionary.
        model_path: Path to saved model.
        data_path: Path to data file. If None, uses the path from config.
        output_dir: Directory to save results.
    """
    print("Starting prediction pipeline...")
    
    # Set up output directory
    output_dir = setup_output_directory(output_dir)
    
    # Initialize components
    preprocessor = DataPreprocessor(config)
    trainer = ModelTrainer(config)
    
    # Load the model
    try:
        model = trainer.load_model(model_path)
        model_name = os.path.basename(model_path).split('_')[0]
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Get data path from config if not provided
    if data_path is None:
        # Get the project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = current_dir
        data_path = os.path.join(project_root, config['data']['test_path'])
    
    # Prepare data
    print(f"Preparing data from {data_path}...")
    try:
        data = preprocessor.load_data(data_path)
        
        # Check if target column exists
        target_column = config['data']['target_column']
        has_target = target_column in data.columns
        
        # Preprocess data
        if has_target:
            X_test, y_test = preprocessor.preprocess_data(data, is_training=False)
            print(f"Data prepared. Test set: {X_test.shape}")
        else:
            # For data without target column
            features = data.drop(columns=config['data']['features_to_exclude'], errors='ignore')
            X_test, _ = preprocessor.preprocess_data(features, is_training=False)
            print(f"Data prepared. Prediction set: {X_test.shape}")
    except Exception as e:
        print(f"Error preparing data: {e}")
        print("Using synthetic data for demonstration...")
        # Create synthetic data for demonstration
        X_test = pd.DataFrame(np.random.randn(50, 10), columns=[f'feature_{i}' for i in range(10)])
        has_target = False
    
    # Make predictions
    print("Making predictions...")
    if isinstance(model, type(keras.Model)):
        # For neural networks
        y_pred_proba = model.predict(X_test)
        if y_pred_proba.shape[1] > 1:  # Multiclass
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:  # Binary
            threshold = config['evaluation']['threshold']
            y_pred = (y_pred_proba > threshold).astype(int)
            y_pred = y_pred.flatten()
    else:
        # For sklearn models
        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        else:
            y_pred_proba = None
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame()
    
    # Add ID column if available
    id_cols = [col for col in data.columns if 'id' in col.lower()]
    if id_cols:
        predictions_df['ID'] = data[id_cols[0]]
    
    # Add predictions
    predictions_df['prediction'] = y_pred
    
    # Add probability columns if available
    if y_pred_proba is not None:
        if isinstance(y_pred_proba, np.ndarray) and y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
            for i in range(y_pred_proba.shape[1]):
                predictions_df[f'probability_class_{i}'] = y_pred_proba[:, i]
        else:
            predictions_df['probability'] = y_pred_proba.flatten()
    
    # Add true values if available
    if has_target:
        predictions_df['true_value'] = y_test
    
    # Save predictions
    predictions_path = os.path.join(output_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")
    
    print(f"Prediction pipeline completed. Results saved to {output_dir}")
    return output_dir

def main():
    """Main function to run the pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run the appropriate pipeline
    if args.mode == 'train':
        train_pipeline(config, args.data, args.output)
    elif args.mode == 'evaluate':
        if args.model_path is None:
            print("Error: Model path is required for evaluate mode.")
            return
        evaluate_pipeline(config, args.model_path, args.data, args.output)
    elif args.mode == 'predict':
        if args.model_path is None:
            print("Error: Model path is required for predict mode.")
            return
        predict_pipeline(config, args.model_path, args.data, args.output)

if __name__ == "__main__":
    main()
