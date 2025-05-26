"""
Script to download and prepare the AI4I 2020 Predictive Maintenance Dataset.
"""
import os
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

def download_dataset(output_dir='data'):
    """
    Download the AI4I 2020 Predictive Maintenance Dataset from UCI ML Repository.
    
    Args:
        output_dir: Directory to save the dataset.
    """
    print("Downloading AI4I 2020 Predictive Maintenance Dataset...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'processed'), exist_ok=True)
    
    # Fetch dataset using UCI ML Repo API
    try:
        # Fetch the dataset
        predictive_maintenance = fetch_ucirepo(id=601)
        
        # Get features and targets
        X = predictive_maintenance.data.features
        y = predictive_maintenance.data.targets
        
        # Combine features and targets
        data = pd.concat([X, y], axis=1)
        
        # Save raw data
        raw_data_path = os.path.join(output_dir, 'raw', 'predictive_maintenance.csv')
        data.to_csv(raw_data_path, index=False)
        print(f"Raw dataset saved to {raw_data_path}")
        
        # Create train and test splits
        from sklearn.model_selection import train_test_split
        
        # Split data into train (70%), validation (15%), and test (15%) sets
        train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42, stratify=data['Machine failure'])
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=temp_data['Machine failure'])
        
        # Save train, validation, and test data
        train_data_path = os.path.join(output_dir, 'processed', 'train.csv')
        val_data_path = os.path.join(output_dir, 'processed', 'validation.csv')
        test_data_path = os.path.join(output_dir, 'processed', 'test.csv')
        
        train_data.to_csv(train_data_path, index=False)
        val_data.to_csv(val_data_path, index=False)
        test_data.to_csv(test_data_path, index=False)
        
        print(f"Training dataset saved to {train_data_path}")
        print(f"Validation dataset saved to {val_data_path}")
        print(f"Test dataset saved to {test_data_path}")
        
        # Create a dataset info file
        dataset_info = {
            'name': 'AI4I 2020 Predictive Maintenance Dataset',
            'source': 'UCI Machine Learning Repository',
            'url': 'https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset',
            'description': 'Synthetic dataset that reflects real predictive maintenance encountered in industry.',
            'instances': len(data),
            'features': len(X.columns),
            'target': 'Machine failure',
            'classes': data['Machine failure'].value_counts().to_dict(),
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data)
        }
        
        # Save dataset info as JSON
        import json
        with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
            json.dump(dataset_info, f, indent=4)
        
        print("Dataset preparation completed successfully!")
        return True
    
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        
        # Create synthetic data if download fails
        print("Creating synthetic data for demonstration...")
        
        # Number of samples
        n_samples = 10000
        
        # Create synthetic features
        synthetic_data = pd.DataFrame({
            'UID': range(1, n_samples + 1),
            'Product ID': np.random.choice(['L', 'M', 'H'], size=n_samples, p=[0.5, 0.3, 0.2]).astype(str) + 
                          np.random.randint(1, 10000, size=n_samples).astype(str),
            'Type': np.random.choice(['L', 'M', 'H'], size=n_samples, p=[0.5, 0.3, 0.2]),
            'Air temperature [K]': np.random.normal(300, 2, n_samples),
            'Process temperature [K]': np.random.normal(310, 1, n_samples),
            'Rotational speed [rpm]': np.random.normal(1500, 100, n_samples),
            'Torque [Nm]': np.maximum(0, np.random.normal(40, 10, n_samples)),
            'Tool wear [min]': np.random.uniform(0, 250, n_samples)
        })
        
        # Create failure indicators based on the dataset description
        twf = (synthetic_data['Tool wear [min]'] > 200).astype(int)
        hdf = ((synthetic_data['Air temperature [K]'] - synthetic_data['Process temperature [K]']).abs() < 8.6) & \
              (synthetic_data['Rotational speed [rpm]'] < 1380)
        hdf = hdf.astype(int)
        
        power = synthetic_data['Rotational speed [rpm]'] * synthetic_data['Torque [Nm]'] * (2 * np.pi / 60)
        pwf = ((power < 3500) | (power > 9000)).astype(int)
        
        tool_wear_torque = synthetic_data['Tool wear [min]'] * synthetic_data['Torque [Nm]']
        threshold = synthetic_data['Type'].map({'L': 11000, 'M': 12000, 'H': 13000})
        osf = (tool_wear_torque > threshold).astype(int)
        
        rnf = (np.random.random(n_samples) < 0.001).astype(int)
        
        # Create machine failure target
        machine_failure = ((twf + hdf + pwf + osf + rnf) > 0).astype(int)
        
        # Add failure indicators and target to the dataframe
        synthetic_data['TWF'] = twf
        synthetic_data['HDF'] = hdf
        synthetic_data['PWF'] = pwf
        synthetic_data['OSF'] = osf
        synthetic_data['RNF'] = rnf
        synthetic_data['Machine failure'] = machine_failure
        
        # Save raw data
        raw_data_path = os.path.join(output_dir, 'raw', 'predictive_maintenance.csv')
        synthetic_data.to_csv(raw_data_path, index=False)
        print(f"Synthetic raw dataset saved to {raw_data_path}")
        
        # Create train and test splits
        from sklearn.model_selection import train_test_split
        
        # Split data into train (70%), validation (15%), and test (15%) sets
        train_data, temp_data = train_test_split(synthetic_data, test_size=0.3, random_state=42, stratify=synthetic_data['Machine failure'])
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=temp_data['Machine failure'])
        
        # Save train, validation, and test data
        train_data_path = os.path.join(output_dir, 'processed', 'train.csv')
        val_data_path = os.path.join(output_dir, 'processed', 'validation.csv')
        test_data_path = os.path.join(output_dir, 'processed', 'test.csv')
        
        train_data.to_csv(train_data_path, index=False)
        val_data.to_csv(val_data_path, index=False)
        test_data.to_csv(test_data_path, index=False)
        
        print(f"Training dataset saved to {train_data_path}")
        print(f"Validation dataset saved to {val_data_path}")
        print(f"Test dataset saved to {test_data_path}")
        
        # Create a dataset info file
        dataset_info = {
            'name': 'AI4I 2020 Predictive Maintenance Dataset (Synthetic)',
            'source': 'Generated locally based on UCI dataset description',
            'description': 'Synthetic dataset that reflects real predictive maintenance encountered in industry.',
            'instances': len(synthetic_data),
            'features': len(synthetic_data.columns) - 6,  # Excluding the 5 failure types and the target
            'target': 'Machine failure',
            'classes': {
                '0': (synthetic_data['Machine failure'] == 0).sum(),
                '1': (synthetic_data['Machine failure'] == 1).sum()
            },
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data)
        }
        
        # Save dataset info as JSON
        import json
        with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
            json.dump(dataset_info, f, indent=4)
        
        print("Synthetic dataset preparation completed successfully!")
        return True

if __name__ == "__main__":
    download_dataset()
