import os
import pandas as pd
import json
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models.usage_predictor import UsagePredictor
from ml_models.cost_predictor import CostPredictor
from ml_models.anomaly_detector import AnomalyDetector

class PredictionService:
    def __init__(self):
        """Initialize the prediction service"""
        self.usage_predictor = UsagePredictor()
        self.cost_predictor = CostPredictor()
        self.anomaly_detector = AnomalyDetector()
        
        # Load historical data - only use for existing bills, not current prediction
        self.historical_data = self._load_historical_data()
    
    def predict_future_bills(self, account_number, months=3):
        """
        Predict future bills for an account
        
        Args:
            account_number: Account number to predict for
            months: Number of months to predict
            
        Returns:
            Dictionary with predictions
        """
        if self.historical_data is None:
            return {'error': 'No historical data available'}
        
        # Filter data for this account
        account_data = self.historical_data[self.historical_data['account_number'] == account_number]
        
        if account_data.empty:
            return {'error': f'No data found for account {account_number}'}
        
        # Generate usage predictions
        usage_predictions = self.usage_predictor.predict(account_data, future_months=months)
        
        if usage_predictions is None:
            return {'error': 'Failed to generate usage predictions'}
        
        # Generate cost predictions for each usage prediction
        predictions = []
        
        for _, row in usage_predictions.iterrows():
            prediction_date = row['prediction_date']
            kwh_prediction = row['predicted_kwh']
            lower_bound = row['lower_bound']
            upper_bound = row['upper_bound']
            
            # Get cost prediction
            cost_data = self.cost_predictor.predict_cost(kwh_prediction)
            if cost_data:
                prediction = {
                    'prediction_date': prediction_date,
                    'predicted_kwh': kwh_prediction,
                    'kwh_lower_bound': lower_bound,
                    'kwh_upper_bound': upper_bound,
                    **cost_data
                }
                predictions.append(prediction)
        
        return {
            'account_number': account_number,
            'predictions': predictions
        }
    
    def detect_bill_anomalies(self, bill_data):
        """
        Detect anomalies in a bill
        
        Args:
            bill_data: Dictionary with bill data
            
        Returns:
            List of anomalies detected
        """
        return self.anomaly_detector.detect_anomalies(bill_data)
    
    def _load_historical_data(self):
        """Load the historical bill data"""
        try:
            data_path = 'data/processed/combined_bills.json'
            
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            
            # Convert date columns
            date_columns = ['bill_date', 'billing_start_date', 'billing_end_date', 'due_date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            return df
        except Exception as e:
            print(f"Error loading historical data: {e}")
            return None
            
    def predict_from_single_bill(self, bill_data, future_months=3):
        """
        Make predictions based only on a single bill without relying on historical data
        
        Args:
            bill_data: Dictionary with current bill data
            future_months: Number of months to predict
            
        Returns:
            List of prediction dictionaries
        """
        try:
            # Create a DataFrame from just this bill
            df = pd.DataFrame([bill_data])
            
            # Convert date columns to datetime
            date_columns = ['bill_date', 'billing_start_date', 'billing_end_date', 'due_date']
            for col in date_columns:
                if col in df.columns and df[col].iloc[0]:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Get usage predictions
            usage_predictions = self.usage_predictor.predict(df, future_months=future_months)
            
            if usage_predictions is None:
                print("Failed to generate usage predictions from single bill")
                return []
            
            predictions = []
            for _, row in usage_predictions.iterrows():
                kwh_prediction = row['predicted_kwh']
                
                # Calculate the cost based on predicted kWh
                cost_prediction = self.cost_predictor.predict_cost(kwh_prediction)
                
                if cost_prediction:
                    predictions.append({
                        "prediction_date": row['prediction_date'],
                        "predicted_kwh": kwh_prediction,
                        "total_bill_amount": cost_prediction['total_bill_amount'],
                        "utility_charges": cost_prediction['utility_charges'],
                        "supplier_charges": cost_prediction['supplier_charges']
                    })
                    
            return predictions
        except Exception as e:
            print(f"Error predicting from single bill: {e}")
            import traceback
            traceback.print_exc()
            return []