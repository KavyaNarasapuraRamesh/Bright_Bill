import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import pickle
import os

class AnomalyDetector:
    def __init__(self, model_dir='models'):
        """Initialize the anomaly detector model"""
        self.model_dir = model_dir
        self.model = None
        self.thresholds = {}
        os.makedirs(model_dir, exist_ok=True)
    
    def train(self, data):
        """
        Train the anomaly detection model
        
        Args:
            data: DataFrame containing historical bill data
        """
        print("Training anomaly detection model...")
        
        # Prepare data
        df = data.copy()
        
        # Calculate statistics for z-score based detection
        for col in ['kwh_used', 'avg_daily_usage', 'total_bill_amount']:
            if col in df.columns:
                self.thresholds[col + '_mean'] = df[col].mean()
                self.thresholds[col + '_std'] = df[col].std()
        
        # Train isolation forest for more complex anomalies
        # Select features for anomaly detection
        features = ['kwh_used', 'avg_daily_usage', 'avg_daily_temperature']
        features = [f for f in features if f in df.columns]
        
        if len(features) > 0:
            X = df[features]
            self.model = IsolationForest(contamination=0.1, random_state=42)
            self.model.fit(X)
        
        # Save model
        self._save_model()
        
        return True
    
    def detect_anomalies(self, bill_data):
        """
        Detect anomalies in a bill
        
        Args:
            bill_data: Dictionary or DataFrame row with bill data
            
        Returns:
            List of anomalies detected
        """
        if not self.thresholds:
            self._load_model()
            if not self.thresholds:
                print("No trained model found. Please train the model first.")
                return []
        
        anomalies = []
        
        # Convert dict to DataFrame if needed
        if isinstance(bill_data, dict):
            df = pd.DataFrame([bill_data])
        else:
            df = pd.DataFrame([bill_data.to_dict()])
        
        # Check for simple anomalies (z-score based)
        for col in ['kwh_used', 'avg_daily_usage', 'total_bill_amount']:
            if col in df.columns and col + '_mean' in self.thresholds and col + '_std' in self.thresholds:
                mean = self.thresholds[col + '_mean']
                std = self.thresholds[col + '_std']
                
                if std > 0:  # Avoid division by zero
                    z_score = abs((df[col].iloc[0] - mean) / std)
                    
                    if z_score > 2:
                        anomalies.append({
                            'type': f'{col}_anomaly',
                            'description': f'{col} value of {df[col].iloc[0]} is unusual (z-score: {z_score:.2f})',
                            'severity': 'high' if z_score > 3 else 'medium'
                        })
        
        # Check for complex anomalies using isolation forest
        if self.model is not None:
            # Select features for prediction
            features = ['kwh_used', 'avg_daily_usage', 'avg_daily_temperature']
            available_features = [f for f in features if f in df.columns]
            
            if len(available_features) > 0:
                # Predict anomaly score
                X = df[available_features]
                prediction = self.model.predict(X)
                
                if prediction[0] == -1:  # Anomaly
                    anomalies.append({
                        'type': 'pattern_anomaly',
                        'description': 'Unusual pattern detected in usage and temperature relationship',
                        'severity': 'medium'
                    })
        
        # Check for calculation anomalies
        if all(col in df.columns for col in ['kwh_used', 'supplier_rate', 'supplier_charges']):
            expected_supplier_charge = df['kwh_used'].iloc[0] * df['supplier_rate'].iloc[0]
            actual_supplier_charge = df['supplier_charges'].iloc[0]
            
            if abs(expected_supplier_charge - actual_supplier_charge) > 1.0:  # More than $1 difference
                anomalies.append({
                    'type': 'calculation_anomaly',
                    'description': f'Supplier charge (${actual_supplier_charge:.2f}) doesn\'t match calculated value (${expected_supplier_charge:.2f})',
                    'severity': 'high'
                })
        
        return anomalies
    
    def _save_model(self):
        """Save the trained model and thresholds"""
        model_path = os.path.join(self.model_dir, 'anomaly_detector_model.pkl')
        thresholds_path = os.path.join(self.model_dir, 'anomaly_detector_thresholds.pkl')
        
        if self.model is not None:
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
        
        with open(thresholds_path, 'wb') as f:
            pickle.dump(self.thresholds, f)
        
        print(f"Anomaly detection model saved to {model_path}")
    
    def _load_model(self):
        """Load the trained model and thresholds"""
        model_path = os.path.join(self.model_dir, 'anomaly_detector_model.pkl')
        thresholds_path = os.path.join(self.model_dir, 'anomaly_detector_thresholds.pkl')
        
        try:
            with open(thresholds_path, 'rb') as f:
                self.thresholds = pickle.load(f)
            
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            except:
                self.model = None
            
            return True
        except FileNotFoundError:
            print(f"Anomaly detection model files not found at {thresholds_path}")
            return False