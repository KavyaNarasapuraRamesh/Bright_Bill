import os
import json
import pandas as pd
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models.usage_predictor import UsagePredictor
from ml_models.cost_predictor import CostPredictor
from ml_models.anomaly_detector import AnomalyDetector

def load_data():
    """Load the combined bill data"""
    try:
        data_path = 'data/processed/combined_bills.json'
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        
        # Convert date columns
        date_columns = ['bill_date', 'billing_start_date', 'billing_end_date', 'due_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def main():
    """Train all prediction models"""
    print("Training prediction models...")
    
    # Load data
    data = load_data()
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    print(f"Loaded {len(data)} bills for training")
    
    # Create output directory
    os.makedirs('data/processed/predictions', exist_ok=True)
    
    # Train usage predictor
    usage_predictor = UsagePredictor()
    usage_predictor.train(data)
    
    # Train cost predictor
    cost_predictor = CostPredictor()
    cost_predictor.train(data)
    
    # Train anomaly detector
    anomaly_detector = AnomalyDetector()
    anomaly_detector.train(data)
    
    print("All models trained successfully!")

if __name__ == "__main__":
    main()