import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import os

class CombinedPredictor:
    """Predictor that combines bill data with appliance usage data"""
    
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        os.makedirs(model_dir, exist_ok=True)
        
        # Load the model if it exists
        self._load_model()
    
    def train(self, data):
        """
        Train the combined prediction model
        
        Args:
            data: DataFrame with both bill data and appliance usage
        """
        print("Training combined prediction model...")
        
        # Prepare data
        df = data.copy()
        
        # Features: both bill data and appliance usage
        features = [
            # Bill data features
            'prev_month_kwh', 'avg_daily_usage', 'avg_daily_temperature',
            # Appliance features
            'air_conditioner_hours', 'refrigerator_hours', 
            'electric_water_heater_hours', 'clothes_dryer_hours', 
            'washing_machine_hours', 'household_size'
        ]
        
        # Use only available features
        available_features = [f for f in features if f in df.columns]
        X = df[available_features]
        
        # Target: total kWh
        y = df['total_kwh']
        
        # Train with gradient boosting (handles missing values well)
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
        
        # Save the model
        self._save_model()
        
        return True
    
    def predict(self, bill_data, appliance_usage):
        """
        Predict electricity usage based on both bill data and appliance usage
        
        Args:
            bill_data: Dictionary with bill information
            appliance_usage: Dictionary with appliance usage hours
            
        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            self._load_model()
            if self.model is None:
                print("No trained model found. Using separate models instead.")
                return None
        
        try:
            # Prepare features
            features = {}
            
            # Bill features
            features['prev_month_kwh'] = bill_data.get('kwh_used', 0)
            features['avg_daily_usage'] = bill_data.get('avg_daily_usage', 0)
            features['avg_daily_temperature'] = bill_data.get('avg_daily_temperature', 70)
            
            # Appliance features
            features['air_conditioner_hours'] = appliance_usage.get('air_conditioner', 0)
            features['refrigerator_hours'] = appliance_usage.get('refrigerator', 24)
            features['electric_water_heater_hours'] = appliance_usage.get('electric_water_heater', 0)
            features['clothes_dryer_hours'] = appliance_usage.get('clothes_dryer', 0)
            features['washing_machine_hours'] = appliance_usage.get('washing_machine', 0)
            features['household_size'] = appliance_usage.get('household_size', 3)
            
            # Convert to DataFrame for prediction
            features_df = pd.DataFrame([features])
            
            # Scale features
            features_scaled = self.scaler.transform(features_df)
            
            # Predict
            predicted_kwh = self.model.predict(features_scaled)[0]
            
            # Calculate costs
            estimated_cost = predicted_kwh * 0.15  # Using average rate
            
            # Distribute among appliances (simplified)
            breakdown = {}
            appliance_data = {
                'air_conditioner': {'avg_wattage': 1500, 'factor': 1.0},
                'refrigerator': {'avg_wattage': 150, 'factor': 24.0},
                'electric_water_heater': {'avg_wattage': 4000, 'factor': 0.9},
                'clothes_dryer': {'avg_wattage': 3000, 'factor': 1.0},
                'washing_machine': {'avg_wattage': 500, 'factor': 1.0}
            }
            
            # Calculate raw energy
            total_energy = 0
            for appliance, hours in appliance_usage.items():
                if appliance in appliance_data and hours > 0:
                    app_info = appliance_data[appliance]
                    energy = (app_info['avg_wattage'] * hours * app_info['factor'] * 30) / 1000
                    total_energy += energy
            
            # Distribute predicted energy proportionally
            if total_energy > 0:
                for appliance, hours in appliance_usage.items():
                    if appliance in appliance_data and hours > 0:
                        app_info = appliance_data[appliance]
                        energy = (app_info['avg_wattage'] * hours * app_info['factor'] * 30) / 1000
                        proportion = energy / total_energy
                        
                        breakdown[appliance] = {
                            'hours_per_day': hours,
                            'monthly_kwh': round(predicted_kwh * proportion, 2),
                            'percentage': round(proportion * 100, 1)
                        }
            
            return {
                'total_kwh': round(predicted_kwh, 2),
                'estimated_cost': round(estimated_cost, 2),
                'breakdown': breakdown
            }
            
        except Exception as e:
            print(f"Error in combined prediction: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None
    
    def _save_model(self):
        """Save the trained model and scaler"""
        model_path = os.path.join(self.model_dir, 'combined_predictor_model.pkl')
        scaler_path = os.path.join(self.model_dir, 'combined_predictor_scaler.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Combined prediction model saved to {model_path}")
    
    def _load_model(self):
        """Load the trained model and scaler"""
        model_path = os.path.join(self.model_dir, 'combined_predictor_model.pkl')
        scaler_path = os.path.join(self.model_dir, 'combined_predictor_scaler.pkl')
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            return True
        except FileNotFoundError:
            print(f"Combined prediction model files not found at {model_path}")
            return False