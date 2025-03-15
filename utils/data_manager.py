# utils/data_manager.py
import os
import json
import pandas as pd
from datetime import datetime

def save_bill_data_to_history(bill_data):
    """Save bill data to a historical dataset for improving predictions"""
    try:
        # Path for historical data
        history_path = 'data/processed/historical_bills.json'
        
        # Load existing historical data or create a new dataset
        try:
            with open(history_path, 'r') as f:
                historical_bills = json.load(f)
                print(f"Loaded {len(historical_bills)} historical bills")
        except (FileNotFoundError, json.JSONDecodeError):
            historical_bills = []
            print("Creating new historical bills dataset")
        
        # Check if we already have this bill in our history (avoid duplicates)
        is_duplicate = False
        for existing_bill in historical_bills:
            # Compare key fields to determine if this is a duplicate
            if (existing_bill.get('account_number') == bill_data.get('account_number') and
                existing_bill.get('bill_date') == bill_data.get('bill_date') and
                existing_bill.get('kwh_used') == bill_data.get('kwh_used')):
                is_duplicate = True
                break
        
        # Only add if it's not a duplicate
        if not is_duplicate:
            historical_bills.append(bill_data)
            
            # Save updated history
            with open(history_path, 'w') as f:
                json.dump(historical_bills, f, indent=2, default=str)
            print(f"Added new bill to historical dataset (total: {len(historical_bills)})")
            
            # Periodically retrain models if we have enough new data
            if len(historical_bills) % 5 == 0:  # Retrain after every 5 new bills
                print("Dataset has grown - scheduling model retraining")
                # This could be a background task for a real app
                # For hackathon, we can use a simple flag file:
                os.makedirs('data/models', exist_ok=True)
                with open('data/models/retrain_needed.txt', 'w') as f:
                    f.write(str(datetime.now()))
        else:
            print("Bill already exists in historical dataset - skipping")
                
        return True
    except Exception as e:
        print(f"Error saving bill to history: {str(e)}")
        return False

def retrain_models_with_history():
    """Retrain prediction models using the accumulated historical data"""
    try:
        # Path for historical data
        history_path = 'data/processed/historical_bills.json'
        
        # Check if we have enough data and if retraining is needed
        try:
            with open(history_path, 'r') as f:
                historical_bills = json.load(f)
            
            if len(historical_bills) < 10:  # Need at least 10 bills for meaningful training
                print("Not enough historical data for retraining")
                return False
        except (FileNotFoundError, json.JSONDecodeError):
            print("No historical data found")
            return False
        
        print(f"Retraining models with {len(historical_bills)} bills")
        
        # Convert to DataFrame
        df = pd.DataFrame(historical_bills)
        
        # Fix date fields
        date_columns = ['bill_date', 'billing_start_date', 'billing_end_date', 'due_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Skip records with missing essential data
        df = df.dropna(subset=['kwh_used', 'total_bill_amount'])
        
        if len(df) < 10:
            print("Not enough valid historical data after cleaning")
            return False
        
        # Retrain each model
        from ml_models.usage_predictor import UsagePredictor
        from ml_models.cost_predictor import CostPredictor
        from ml_models.anomaly_detector import AnomalyDetector
        
        # Initialize and train models
        usage_model = UsagePredictor()
        cost_model = CostPredictor()
        anomaly_model = AnomalyDetector()
        
        print("Training usage prediction model...")
        usage_model.train(df)
        
        print("Training cost prediction model...")
        cost_model.train(df)
        
        print("Training anomaly detection model...")
        anomaly_model.train(df)
        
        # Clear the retrain flag
        if os.path.exists('data/models/retrain_needed.txt'):
            os.remove('data/models/retrain_needed.txt')
            
        print("Model retraining complete!")
        return True
        
    except Exception as e:
        print(f"Error during model retraining: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False