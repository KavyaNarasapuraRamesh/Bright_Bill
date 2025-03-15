import os
import pandas as pd
from ml_models.appliance_predictor import AppliancePredictor
from ml_models.combined_predictor import CombinedPredictor

def train_appliance_model():
    """Train the appliance usage prediction model"""
    # Check if training data exists
    data_path = 'data/processed/appliance_training_data.csv'
    
    if not os.path.exists(data_path):
        print(f"Training data not found at {data_path}. Generating dummy data...")
        from scripts.generate_dummy_data import generate_appliance_data
        df = generate_appliance_data(100)
    else:
        df = pd.read_csv(data_path)
    
    print(f"Training appliance model with {len(df)} samples")
    
    # Initialize and train the model
    predictor = AppliancePredictor()
    success = predictor.train(df)
    
    if success:
        print("Appliance model training complete!")
    else:
        print("Failed to train appliance model")
    
    return success

def train_combined_model():
    """Train the combined bill+appliance prediction model"""
    # Check if training data exists
    data_path = 'data/processed/combined_training_data.csv'
    
    if not os.path.exists(data_path):
        print(f"Training data not found at {data_path}. Generating dummy data...")
        from scripts.generate_dummy_data import generate_combined_data
        df = generate_combined_data(100)
    else:
        df = pd.read_csv(data_path)
    
    print(f"Training combined model with {len(df)} samples")
    
    # Convert date columns to datetime
    if 'bill_date' in df.columns:
        df['bill_date'] = pd.to_datetime(df['bill_date'])
    
    # Initialize and train the model
    predictor = CombinedPredictor()
    success = predictor.train(df)
    
    if success:
        print("Combined model training complete!")
    else:
        print("Failed to train combined model")
    
    return success

if __name__ == "__main__":
    print("Starting model training...")
    
    # Train both models
    appliance_success = train_appliance_model()
    combined_success = train_combined_model()
    
    print("Training complete!")
    print(f"Appliance model: {'Success' if appliance_success else 'Failed'}")
    print(f"Combined model: {'Success' if combined_success else 'Failed'}")