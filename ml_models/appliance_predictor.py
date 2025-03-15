import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class AppliancePredictor:
    def __init__(self, model_dir='models'):
        """Initialize the appliance usage predictor"""
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        os.makedirs(model_dir, exist_ok=True)
        
        self.appliance_data = {
            'air_conditioner': {'avg_wattage': 1500, 'factor': 1.0},
            'refrigerator': {'avg_wattage': 150, 'factor': 24.0},  # Always on
            'electric_water_heater': {'avg_wattage': 4000, 'factor': 0.9},
            'clothes_dryer': {'avg_wattage': 3000, 'factor': 1.0},
            'washing_machine': {'avg_wattage': 500, 'factor': 1.0}
        }
        
        # Load the model if it exists
        self._load_model()
    
    def train(self, data):
        """
        Train the appliance usage prediction model
        
        Args:
            data: DataFrame containing usage data with columns for each appliance and total kWh
        """
        print("Training appliance prediction model...")
        
        # Prepare data
        df = data.copy()
        
        # Features: appliance usage hours and other factors
        X = df[['air_conditioner_hours', 'refrigerator_hours', 
                'electric_water_heater_hours', 'clothes_dryer_hours', 
                'washing_machine_hours', 'household_size', 'home_sqft']]
        
        # Target: total kWh
        y = df['total_kwh']
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train the model - Random Forest works well for this type of data
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
        
        # Save the model
        self._save_model()
        
        return True
    
    def predict(self, appliance_usage, household_size=3, home_sqft=1800):
        """
        Predict total kWh based on appliance usage
        
        Args:
            appliance_usage: Dictionary with appliance names and hours used
            household_size: Number of people in household
            home_sqft: Square footage of home
            
        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            self._load_model()
            if self.model is None:
                # Fall back to simple calculation if no model exists
                return self._simple_prediction(appliance_usage)
        
        try:
            # Create feature array
            features = np.array([[
                appliance_usage.get('air_conditioner', 0),
                appliance_usage.get('refrigerator', 24),  # Default 24h for refrigerator
                appliance_usage.get('electric_water_heater', 0),
                appliance_usage.get('clothes_dryer', 0),
                appliance_usage.get('washing_machine', 0),
                household_size,
                home_sqft
            ]])
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict total kWh
            predicted_kwh = self.model.predict(features_scaled)[0]
            
            # Calculate breakdown based on wattage ratios
            total_energy = 0
            breakdown = {}
            
            for appliance, hours in appliance_usage.items():
                if appliance in self.appliance_data and hours > 0:
                    appliance_info = self.appliance_data[appliance]
                    energy = (appliance_info['avg_wattage'] * hours * appliance_info['factor'] * 30) / 1000  # Monthly kWh
                    total_energy += energy
                    breakdown[appliance] = {
                        'hours_per_day': hours,
                        'monthly_kwh': 0  # Will be adjusted after calculating percentages
                    }
            
            # Adjust breakdown to match predicted total
            if total_energy > 0:
                for appliance in breakdown:
                    appliance_info = self.appliance_data[appliance]
                    hours = appliance_usage[appliance]
                    raw_energy = (appliance_info['avg_wattage'] * hours * appliance_info['factor'] * 30) / 1000
                    percentage = raw_energy / total_energy
                    breakdown[appliance]['monthly_kwh'] = round(predicted_kwh * percentage, 2)
                    breakdown[appliance]['percentage'] = round(percentage * 100, 1)
            
            # Estimate cost (using average rate of $0.15 per kWh)
            estimated_cost = predicted_kwh * 0.15
            
            return {
                'total_kwh': round(predicted_kwh, 2),
                'estimated_cost': round(estimated_cost, 2),
                'breakdown': breakdown
            }
            
        except Exception as e:
            print(f"Error in ML prediction: {str(e)}")
            # Fall back to simple calculation
            return self._simple_prediction(appliance_usage)
    
    def _simple_prediction(self, appliance_usage):
        """Simple prediction based on wattage when ML model is unavailable"""
        total_kwh = 0
        breakdown = {}
        
        # Calculate kWh for each appliance
        for appliance, hours in appliance_usage.items():
            if appliance in self.appliance_data:
                appliance_info = self.appliance_data[appliance]
                # Monthly kWh = (wattage * hours per day * factor * 30 days) / 1000
                kwh = (appliance_info['avg_wattage'] * hours * appliance_info['factor'] * 30) / 1000
                total_kwh += kwh
                breakdown[appliance] = {
                    'hours_per_day': hours,
                    'monthly_kwh': round(kwh, 2),
                    'percentage': 0  # Will be calculated after total is known
                }
        
        # Calculate percentages
        for appliance in breakdown:
            if total_kwh > 0:
                breakdown[appliance]['percentage'] = round((breakdown[appliance]['monthly_kwh'] / total_kwh * 100), 1)
        
        # Estimate cost (using average rate of $0.15 per kWh)
        estimated_cost = total_kwh * 0.15
        
        return {
            'total_kwh': round(total_kwh, 2),
            'estimated_cost': round(estimated_cost, 2),
            'breakdown': breakdown
        }
    
    def _save_model(self):
        """Save the trained model and scaler"""
        model_path = os.path.join(self.model_dir, 'appliance_predictor_model.pkl')
        scaler_path = os.path.join(self.model_dir, 'appliance_predictor_scaler.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Appliance prediction model saved to {model_path}")
    
    def _load_model(self):
        """Load the trained model and scaler"""
        model_path = os.path.join(self.model_dir, 'appliance_predictor_model.pkl')
        scaler_path = os.path.join(self.model_dir, 'appliance_predictor_scaler.pkl')
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            return True
        except FileNotFoundError:
            print(f"Appliance prediction model files not found at {model_path}")
            return False