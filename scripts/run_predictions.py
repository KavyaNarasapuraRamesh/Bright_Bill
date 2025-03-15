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
    """Run predictions on the data"""
    print("Running predictions...")
    
    # Load data
    data = load_data()
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    # Create predictions directory
    os.makedirs('data/processed/predictions', exist_ok=True)
    
    # Run usage predictions
    usage_predictor = UsagePredictor()
    usage_predictions = usage_predictor.predict(data, future_months=6)
    
    if usage_predictions is not None:
        # Save usage predictions
        usage_predictions.to_csv('data/processed/predictions/usage_predictions.csv', index=False)
        usage_predictions.to_json('data/processed/predictions/usage_predictions.json', orient='records', date_format='iso')
        print(f"Saved usage predictions for {len(usage_predictions)} months")
        
        # Generate cost predictions for each usage prediction
        cost_predictor = CostPredictor()
        
        cost_predictions = []
        for _, row in usage_predictions.iterrows():
            prediction_date = row['prediction_date']
            kwh_prediction = row['predicted_kwh']
            
            # Get cost prediction
            cost_data = cost_predictor.predict_cost(kwh_prediction)
            if cost_data:
                cost_data['prediction_date'] = prediction_date
                cost_data['predicted_kwh'] = kwh_prediction
                cost_predictions.append(cost_data)
        
        # Save cost predictions
        if cost_predictions:
            cost_df = pd.DataFrame(cost_predictions)
            cost_df.to_csv('data/processed/predictions/cost_predictions.csv', index=False)
            cost_df.to_json('data/processed/predictions/cost_predictions.json', orient='records', date_format='iso')
            print(f"Saved cost predictions for {len(cost_predictions)} months")
    
    # Run anomaly detection on the latest bill
    anomaly_detector = AnomalyDetector()
    latest_bill = data.sort_values('bill_date').iloc[-1]
    anomalies = anomaly_detector.detect_anomalies(latest_bill)
    
    if anomalies:
        # Add bill_id to each anomaly
        for anomaly in anomalies:
            anomaly['bill_date'] = latest_bill['bill_date']
        
        # Save anomalies
        anomalies_df = pd.DataFrame(anomalies)
        anomalies_df.to_csv('data/processed/predictions/anomalies.csv', index=False)
        anomalies_df.to_json('data/processed/predictions/anomalies.json', orient='records', date_format='iso')
        print(f"Saved {len(anomalies)} detected anomalies")
    else:
        print("No anomalies detected in the latest bill")
    
    print("All predictions completed!")

if __name__ == "__main__":
    main()