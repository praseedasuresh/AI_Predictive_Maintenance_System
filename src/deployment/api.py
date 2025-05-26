"""
API module for deploying the predictive maintenance model.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import joblib
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import logging
import traceback

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.config_loader import load_config
from src.data_processing.preprocessor import DataPreprocessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()

# Initialize FastAPI app
app = FastAPI(
    title="Predictive Maintenance API",
    description="API for predicting machine failures in industrial equipment",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input data model
class MachineData(BaseModel):
    """
    Input data model for machine sensor readings.
    """
    air_temperature: float = Field(..., description="Air temperature in Kelvin", example=300.0)
    process_temperature: float = Field(..., description="Process temperature in Kelvin", example=310.5)
    rotational_speed: float = Field(..., description="Rotational speed in RPM", example=1500)
    torque: float = Field(..., description="Torque in Nm", example=40.0)
    tool_wear: float = Field(..., description="Tool wear in minutes", example=120.0)
    machine_type: str = Field(..., description="Machine type (L, M, or H)", example="M")

class BatchPredictionRequest(BaseModel):
    """
    Batch prediction request model.
    """
    machines: List[MachineData]

class PredictionResponse(BaseModel):
    """
    Prediction response model.
    """
    machine_failure: bool
    failure_probability: float
    prediction_time: str
    failure_type: Dict[str, float] = None
    maintenance_recommendation: str = None

class BatchPredictionResponse(BaseModel):
    """
    Batch prediction response model.
    """
    predictions: List[PredictionResponse]
    summary: Dict[str, Any]

# Global variables for model and preprocessor
model = None
preprocessor = None
prediction_history = []

def load_model():
    """
    Load the trained model.
    
    Returns:
        Loaded model.
    """
    global model
    if model is None:
        model_path = config['deployment']['pipeline_path'] if 'pipeline_path' in config['deployment'] else "models/best_model.pkl"
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Look for any model file in the models directory
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
            if model_files:
                model_path = os.path.join(models_dir, model_files[0])
                model = joblib.load(model_path)
                logger.info(f"Loaded alternative model from {model_path}")
            else:
                raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    return model

def get_preprocessor():
    """
    Get the data preprocessor.
    
    Returns:
        Data preprocessor.
    """
    global preprocessor
    if preprocessor is None:
        preprocessor = DataPreprocessor(config)
        logger.info("Preprocessor initialized")
    return preprocessor

def record_prediction(prediction_data: Dict[str, Any]):
    """
    Record prediction for monitoring.
    
    Args:
        prediction_data: Prediction data to record.
    """
    global prediction_history
    prediction_history.append(prediction_data)
    
    # Keep only the last 1000 predictions
    if len(prediction_history) > 1000:
        prediction_history = prediction_history[-1000:]

def get_maintenance_recommendation(failure_probability: float, failure_type: Dict[str, float]) -> str:
    """
    Generate maintenance recommendation based on prediction.
    
    Args:
        failure_probability: Probability of failure.
        failure_type: Dictionary of failure type probabilities.
        
    Returns:
        Maintenance recommendation.
    """
    if failure_probability < 0.2:
        return "No maintenance required at this time. Continue regular monitoring."
    elif failure_probability < 0.5:
        # Determine the most likely failure type
        most_likely_failure = max(failure_type.items(), key=lambda x: x[1])
        
        if most_likely_failure[0] == 'TWF':
            return "Schedule tool inspection within the next 48 hours. Tool wear is approaching critical levels."
        elif most_likely_failure[0] == 'HDF':
            return "Check cooling system and improve heat dissipation. Monitor temperature differential."
        elif most_likely_failure[0] == 'PWF':
            return "Inspect power system and adjust rotational speed/torque parameters to optimal range."
        elif most_likely_failure[0] == 'OSF':
            return "Reduce operational load or replace tool to prevent overstrain failure."
        else:
            return "Schedule preventive maintenance within the next week."
    else:
        return "URGENT: Immediate maintenance required. High risk of machine failure detected."

@app.on_event("startup")
async def startup_event():
    """
    Initialize components on startup.
    """
    try:
        load_model()
        get_preprocessor()
        logger.info("API started successfully")
    except Exception as e:
        logger.error(f"Error during startup: {e}")

@app.get("/")
async def root():
    """
    Root endpoint.
    """
    return {
        "message": "Predictive Maintenance API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_initialized": preprocessor is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    machine_data: MachineData,
    background_tasks: BackgroundTasks,
    model=Depends(load_model),
    preprocessor=Depends(get_preprocessor)
):
    """
    Predict machine failure from sensor data.
    """
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame({
            'Type': [machine_data.machine_type],
            'Air temperature [K]': [machine_data.air_temperature],
            'Process temperature [K]': [machine_data.process_temperature],
            'Rotational speed [rpm]': [machine_data.rotational_speed],
            'Torque [Nm]': [machine_data.torque],
            'Tool wear [min]': [machine_data.tool_wear]
        })
        
        # Make prediction
        prediction_time = datetime.now().isoformat()
        
        # Get failure probability
        if hasattr(model, 'predict_proba'):
            failure_proba = model.predict_proba(input_data)[0, 1]
        else:
            # If model doesn't support predict_proba, use a dummy probability
            failure_proba = float(model.predict(input_data)[0])
        
        # Get binary prediction
        failure_prediction = bool(model.predict(input_data)[0])
        
        # Estimate failure type probabilities
        # This is a simplified approach - in a real system, you would have a model for each failure type
        failure_type = {}
        
        # Tool wear failure (TWF)
        if machine_data.tool_wear > 200:
            failure_type['TWF'] = min(0.9, machine_data.tool_wear / 250)
        else:
            failure_type['TWF'] = max(0.1, machine_data.tool_wear / 250)
            
        # Heat dissipation failure (HDF)
        temp_diff = machine_data.process_temperature - machine_data.air_temperature
        if temp_diff < 8.6 and machine_data.rotational_speed < 1380:
            failure_type['HDF'] = 0.8
        else:
            failure_type['HDF'] = 0.2
            
        # Power failure (PWF)
        power = machine_data.rotational_speed * machine_data.torque * (2 * np.pi / 60)
        if power < 3500 or power > 9000:
            failure_type['PWF'] = 0.7
        else:
            failure_type['PWF'] = 0.1
            
        # Overstrain failure (OSF)
        tool_wear_torque = machine_data.tool_wear * machine_data.torque
        threshold = {'L': 11000, 'M': 12000, 'H': 13000}.get(machine_data.machine_type, 12000)
        if tool_wear_torque > threshold:
            failure_type['OSF'] = 0.8
        else:
            failure_type['OSF'] = 0.2
            
        # Random failure (RNF)
        failure_type['RNF'] = 0.001
        
        # Normalize probabilities to sum to failure_proba
        total = sum(failure_type.values())
        for key in failure_type:
            failure_type[key] = (failure_type[key] / total) * failure_proba
        
        # Generate maintenance recommendation
        maintenance_recommendation = get_maintenance_recommendation(failure_proba, failure_type)
        
        # Prepare response
        response = {
            "machine_failure": failure_prediction,
            "failure_probability": failure_proba,
            "prediction_time": prediction_time,
            "failure_type": failure_type,
            "maintenance_recommendation": maintenance_recommendation
        }
        
        # Record prediction in background
        prediction_record = {
            "input": machine_data.dict(),
            "output": response,
            "timestamp": prediction_time
        }
        background_tasks.add_task(record_prediction, prediction_record)
        
        return response
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    batch_request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    model=Depends(load_model),
    preprocessor=Depends(get_preprocessor)
):
    """
    Make batch predictions for multiple machines.
    """
    try:
        predictions = []
        
        for machine_data in batch_request.machines:
            # Convert input data to DataFrame
            input_data = pd.DataFrame({
                'Type': [machine_data.machine_type],
                'Air temperature [K]': [machine_data.air_temperature],
                'Process temperature [K]': [machine_data.process_temperature],
                'Rotational speed [rpm]': [machine_data.rotational_speed],
                'Torque [Nm]': [machine_data.torque],
                'Tool wear [min]': [machine_data.tool_wear]
            })
            
            # Make prediction
            prediction_time = datetime.now().isoformat()
            
            # Get failure probability
            if hasattr(model, 'predict_proba'):
                failure_proba = model.predict_proba(input_data)[0, 1]
            else:
                # If model doesn't support predict_proba, use a dummy probability
                failure_proba = float(model.predict(input_data)[0])
            
            # Get binary prediction
            failure_prediction = bool(model.predict(input_data)[0])
            
            # Estimate failure type probabilities (simplified)
            failure_type = {}
            
            # Tool wear failure (TWF)
            if machine_data.tool_wear > 200:
                failure_type['TWF'] = min(0.9, machine_data.tool_wear / 250)
            else:
                failure_type['TWF'] = max(0.1, machine_data.tool_wear / 250)
                
            # Heat dissipation failure (HDF)
            temp_diff = machine_data.process_temperature - machine_data.air_temperature
            if temp_diff < 8.6 and machine_data.rotational_speed < 1380:
                failure_type['HDF'] = 0.8
            else:
                failure_type['HDF'] = 0.2
                
            # Power failure (PWF)
            power = machine_data.rotational_speed * machine_data.torque * (2 * np.pi / 60)
            if power < 3500 or power > 9000:
                failure_type['PWF'] = 0.7
            else:
                failure_type['PWF'] = 0.1
                
            # Overstrain failure (OSF)
            tool_wear_torque = machine_data.tool_wear * machine_data.torque
            threshold = {'L': 11000, 'M': 12000, 'H': 13000}.get(machine_data.machine_type, 12000)
            if tool_wear_torque > threshold:
                failure_type['OSF'] = 0.8
            else:
                failure_type['OSF'] = 0.2
                
            # Random failure (RNF)
            failure_type['RNF'] = 0.001
            
            # Normalize probabilities
            total = sum(failure_type.values())
            for key in failure_type:
                failure_type[key] = (failure_type[key] / total) * failure_proba
            
            # Generate maintenance recommendation
            maintenance_recommendation = get_maintenance_recommendation(failure_proba, failure_type)
            
            # Prepare prediction
            prediction = {
                "machine_failure": failure_prediction,
                "failure_probability": failure_proba,
                "prediction_time": prediction_time,
                "failure_type": failure_type,
                "maintenance_recommendation": maintenance_recommendation
            }
            
            predictions.append(prediction)
            
            # Record prediction in background
            prediction_record = {
                "input": machine_data.dict(),
                "output": prediction,
                "timestamp": prediction_time
            }
            background_tasks.add_task(record_prediction, prediction_record)
        
        # Calculate summary statistics
        num_failures = sum(1 for p in predictions if p["machine_failure"])
        failure_rate = num_failures / len(predictions) if predictions else 0
        avg_probability = sum(p["failure_probability"] for p in predictions) / len(predictions) if predictions else 0
        
        # Identify most common failure types
        failure_type_counts = {}
        for p in predictions:
            if p["machine_failure"]:
                most_likely_type = max(p["failure_type"].items(), key=lambda x: x[1])[0]
                failure_type_counts[most_likely_type] = failure_type_counts.get(most_likely_type, 0) + 1
        
        most_common_failure = max(failure_type_counts.items(), key=lambda x: x[1])[0] if failure_type_counts else None
        
        summary = {
            "total_machines": len(predictions),
            "failure_count": num_failures,
            "failure_rate": failure_rate,
            "average_failure_probability": avg_probability,
            "most_common_failure_type": most_common_failure,
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "predictions": predictions,
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Error making batch prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/monitoring/stats")
async def get_monitoring_stats():
    """
    Get monitoring statistics.
    """
    if not prediction_history:
        return {
            "message": "No predictions recorded yet",
            "count": 0
        }
    
    # Calculate statistics
    total_predictions = len(prediction_history)
    failure_predictions = sum(1 for p in prediction_history if p["output"]["machine_failure"])
    failure_rate = failure_predictions / total_predictions if total_predictions > 0 else 0
    
    # Get average probabilities
    avg_probability = sum(p["output"]["failure_probability"] for p in prediction_history) / total_predictions
    
    # Get failure type distribution
    failure_types = {}
    for p in prediction_history:
        if p["output"]["machine_failure"]:
            failure_type = max(p["output"]["failure_type"].items(), key=lambda x: x[1])[0]
            failure_types[failure_type] = failure_types.get(failure_type, 0) + 1
    
    # Get recent trend (last 100 predictions)
    recent = prediction_history[-100:] if len(prediction_history) >= 100 else prediction_history
    recent_failure_rate = sum(1 for p in recent if p["output"]["machine_failure"]) / len(recent)
    
    return {
        "total_predictions": total_predictions,
        "failure_predictions": failure_predictions,
        "failure_rate": failure_rate,
        "average_failure_probability": avg_probability,
        "failure_type_distribution": failure_types,
        "recent_failure_rate": recent_failure_rate,
        "timestamp": datetime.now().isoformat()
    }

def start_server():
    """
    Start the API server.
    """
    port = config['deployment']['api_port'] if 'api_port' in config['deployment'] else 8000
    uvicorn.run("src.deployment.api:app", host="0.0.0.0", port=port, reload=True)

if __name__ == "__main__":
    start_server()
