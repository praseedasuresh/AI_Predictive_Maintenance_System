# AI Predictive Maintenance System

A comprehensive machine learning solution for predicting industrial equipment failures before they occur, enabling proactive maintenance scheduling and reducing costly downtime.

(Note : This is a demo version of the system. It is not intended for production use. Only I created for my inernship purpose)

## Project Overview

This project implements an end-to-end predictive maintenance system using machine learning to forecast when industrial equipment is likely to fail. The system analyzes sensor data from machines to identify patterns that precede failures, allowing maintenance teams to intervene before breakdowns occur.

Key business benefits:
- Reduce unplanned downtime by up to 50%
- Extend equipment lifespan by 20-40%
- Lower maintenance costs by 10-30% 
- Optimize spare parts inventory



## Challenges in AI Predictive Maintenance Systems

### Technical Challenges

- Data Quality and Availability:
Industrial equipment often lacks sufficient sensors to capture all relevant parameters
Historical failure data is typically imbalanced (few failure examples compared to normal operation)
Sensor data may contain noise, missing values, or drift over time

- Feature Engineering Complexity:
Domain-specific knowledge is required to create meaningful features
Different failure modes require different engineered features
Temporal patterns and degradation signals can be subtle and complex

- Model Generalization:
Models trained on one type of equipment may not generalize to others
Operating conditions vary across facilities and environments
New failure modes may emerge that weren't in the training data

- Real-time Implementation:
Processing sensor data in real-time requires efficient algorithms
Edge computing may be needed for remote equipment
Balancing prediction accuracy with computational efficiency

- Integration Challenges:
Connecting with existing industrial control systems and SCADA
Interfacing with enterprise asset management systems
Ensuring security in industrial networks

### Real-World Applications

- Manufacturing Industry

1.Automotive Assembly Lines: Predict failures in robotic arms and conveyor systems
2.Semiconductor Fabrication: Monitor precision equipment to prevent costly wafer defects
3.Steel Production: Predict furnace failures and optimize maintenance schedules 

-Energy Sector
1.Wind Turbines: Remote monitoring of offshore turbines to predict gearbox and bearing failures
2.Power Plants: Predict failures in generators, turbines, and cooling systems
3.Oil & Gas: Monitor drilling equipment and pipeline integrity

-Transportation
1.Railway Systems: Predict track and train component failures before they cause delays
2.Commercial Aircraft: Monitor engine health and predict maintenance needs
3.Fleet Management: Optimize vehicle maintenance schedules based on actual usage patterns

-Infrastructure
1.HVAC Systems: Predict failures in commercial building climate control systems
2.Water Treatment: Monitor pumps and filtration systems to prevent service interruptions
3.Telecommunications: Predict failures in network infrastructure equipment


### Real-World Success Stories

-Siemens: Implemented predictive maintenance for their gas turbines, reducing unplanned downtime by 30% and saving in maintenance costs.
-Deutsche Bahn: Applied predictive maintenance to their railway switches, reducing failures by 25% and improving on-time performance.
-Rolls-Royce: Uses predictive maintenance for aircraft engines, allowing airlines to schedule maintenance based on actual engine condition rather than fixed intervals.
-Intel: Implemented predictive maintenance in semiconductor manufacturing, reducing equipment downtime by 25% and increasing yield.
-ExxonMobil: Uses predictive maintenance in refineries to prevent unexpected shutdowns, saving an estimated $20 million annually at a single facility.
The value of predictive maintenance increases dramatically in industries where:


## Core Features

- **Real-time Failure Prediction**: Continuously monitor equipment and predict failures with high accuracy
- **Multi-class Failure Classification**: Identify specific types of failures (e.g., tool wear, heat dissipation, power)
- **Automated Feature Engineering**: Extract domain-specific features from raw sensor data
- **Interactive Dashboard**: Monitor equipment health and predictions through an intuitive web interface
- **REST API**: Integrate predictions with existing maintenance management systems
- **Batch Processing**: Process historical data to identify long-term trends
- **Robust Error Handling**: Ensure system reliability with automatic recovery mechanisms

## System Architecture

```
├── config.yaml                 # Central configuration file
├── requirements.txt            # Python dependencies
├── data/                       # Data directory
│   ├── raw/                    # Raw sensor data
│   ├── processed/              # Processed datasets
│   └── splits/                 # Train/validation/test splits
├── models/                     # Trained ML models
├── results/                    # Evaluation metrics and visualizations
├── logs/                       # Application logs
└── src/                        # Source code
    ├── data_processing/        # Data processing pipeline
    │   ├── download_dataset.py # Dataset acquisition
    │   ├── preprocessor.py     # Data preprocessing
    │   └── feature_engineering.py # Feature extraction and engineering
    ├── model/                  # Model development
    │   ├── model_trainer.py    # Model training
    │   └── model_evaluator.py  # Model evaluation
    ├── deployment/             # Deployment components
    │   ├── api.py              # FastAPI REST API
    │   ├── web_ui.py           # Streamlit web interface
    │   └── run_app.py          # Application orchestrator
    └── utils/                  # Utility functions
        ├── config_loader.py    # Configuration management
        ├── setup_directories.py # Project structure setup
        └── demo_setup.py       # Demo data generation and setup
```

## Technical Implementation

### Machine Learning Pipeline

1. **Data Preprocessing**:
   - Standardization of numerical features
   - One-hot encoding of categorical variables
   - Handling of missing values and outliers

2. **Feature Engineering**:
   - Temperature differential calculation
   - Power consumption estimation
   - Tool wear rate normalization
   - Critical operation indicators
   - Time-based aggregation features

3. **Model Training**:
   - Random Forest classifier (primary model)
   - XGBoost ensemble (alternative model)
   - Hyperparameter optimization via grid search
   - Cross-validation for robust performance estimation

4. **Deployment System**:
   - FastAPI backend with automatic documentation
   - Streamlit frontend for interactive visualization
   - Process monitoring with automatic restart capabilities
   - Comprehensive error handling and logging

## Installation and Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/predictive-maintenance.git
cd predictive-maintenance
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up the project structure:
```bash
python -m src.utils.setup_directories
```

4. Generate demo data and train initial model:
```bash
python -m src.utils.demo_setup
```

## Running the Application

Start the complete application (API + Web UI):

```bash
python -m src.deployment.run_app
```

This will:
1. Verify that a trained model exists (or create one if needed)
2. Start the FastAPI server on port 8000
3. Launch the Streamlit web UI on port 8501
4. Open browser tabs for both interfaces

## API Documentation

The REST API provides the following endpoints:

- `GET /health`: System health check
- `POST /predict`: Single machine prediction
- `POST /predict/batch`: Multiple machine predictions
- `GET /monitoring/stats`: System performance statistics

Example prediction request:
```json
{
  "machine_type": "M",
  "air_temperature": 298.5,
  "process_temperature": 308.7,
  "rotational_speed": 1503,
  "torque": 42.8,
  "tool_wear": 189.2
}
```

## Web UI Guide

The web interface provides:

1. **Dashboard**: Equipment health overview with failure distribution charts
2. **Single Prediction**: Interactive form for predicting individual machine failures
3. **Batch Prediction**: Upload CSV files for processing multiple machines
4. **Monitoring**: System performance metrics and prediction history

## Dataset Information

The system uses the AI4I 2020 Predictive Maintenance Dataset, which includes:

- Machine type (L, M, H)
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

Target variables:
- Machine failure (binary)
- Failure type:
  - Tool Wear Failure (TWF)
  - Heat Dissipation Failure (HDF)
  - Power Failure (PWF)
  - Overstrain Failure (OSF)
  - Random Failure (RNF)

## Performance Metrics

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 97.8% | 96.2% | 94.5% | 95.3% |
| XGBoost | 98.1% | 97.3% | 95.1% | 96.2% |

## Troubleshooting

If you encounter issues:

1. **API fails to start**: 
   - Check if port 8000 is already in use
   - Verify all dependencies are installed
   - Check logs for specific error messages

2. **Web UI not loading**:
   - Ensure the API server is running properly
   - Check if port 8501 is available
   - Restart the application with `python -m src.deployment.run_app`

3. **Model errors**:
   - Regenerate the model with `python -m src.utils.demo_setup`
   - Check if the models directory exists and has proper permissions

4. **Data processing errors**:
   - Ensure input data matches the expected format
   - Verify that feature engineering configuration is correct

## Future Development

- Integration with IoT sensor networks for direct data ingestion
- Advanced deep learning models for time series prediction
- Anomaly detection for novel failure modes
- Reinforcement learning for maintenance scheduling optimization
- Mobile application for on-the-go monitoring

## Author

Praseeda G Suresh
