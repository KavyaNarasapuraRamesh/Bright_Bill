import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
import pickle
import os
from datetime import datetime, timedelta

class UsagePredictor:
    def __init__(self, model_dir='models'):
        """Initialize the usage predictor model"""
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        os.makedirs(model_dir, exist_ok=True)
    
    def train(self, data, month_lookahead=1):
        """
        Train the model to predict electricity usage with improved features
        
        Args:
            data: DataFrame containing historical bill data
            month_lookahead: How many months ahead to predict
        """
        print(f"Training usage prediction model for {month_lookahead} month(s) ahead...")
        
        # Prepare data
        df = self._prepare_data(data)
        
        # Create improved features for prediction
        features = self._engineer_enhanced_features(df, month_lookahead)
        
        if features.empty:
            print("Not enough data for training. Need at least 6 months of bills.")
            return False
        
        # Select features and target
        X = features.drop(columns=['target_kwh'])
        
        # Apply log transformation to target for better handling of high variance
        y = np.log1p(features['target_kwh'])
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train XGBoost model instead of RandomForest
        from xgboost import XGBRegressor
        self.model = XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
        self.model.fit(X_scaled, y)
        
        # Save model
        self._save_model()
    
        return True
    
    def predict(self, data, future_months=3):
        """
        Predict usage for future months with improved features
        
        Args:
            data: DataFrame containing historical bill data
            future_months: Number of months to predict
                
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            self._load_model()
            if self.model is None:
                print("No trained model found. Please train the model first.")
                return None
        
        # Prepare data
        df = self._prepare_data(data)
        
        # Get last date in data
        last_date = df['bill_date'].max()
        
        # Create prediction dates
        prediction_dates = [last_date + timedelta(days=30*i) for i in range(1, future_months+1)]
        
        # Create predictions DataFrame
        predictions = []
        
        for pred_date in prediction_dates:
            # Create feature row for this prediction date
            month = pred_date.month
            
            # Get average temperature for this month from historical data
            month_temps = df[df['bill_date'].dt.month == month]['avg_daily_temperature']
            avg_temp = month_temps.mean() if not month_temps.empty else self._default_temp_for_month(month)
            
            # Get the most recent 3 months of data
            recent_data = df.sort_values('bill_date').tail(3)
            
            # Create enhanced features
            features = {
                'month': month,
                'month_sin': np.sin(2 * np.pi * month / 12),
                'month_cos': np.cos(2 * np.pi * month / 12),
                'avg_daily_temperature': avg_temp,
                'days_in_billing_period': 30,  # Standard assumption
                'avg_temp_x_month': avg_temp * month,
                'days_x_temp': 30 * avg_temp,
                'avg_3m_kwh': recent_data['kwh_used'].mean(),
                'last_month_kwh': recent_data.iloc[-1]['kwh_used'] if not recent_data.empty else 0,
                'last_2_month_kwh': recent_data.iloc[-2]['kwh_used'] if len(recent_data) > 1 else 0,
                'last_3_month_kwh': recent_data.iloc[-3]['kwh_used'] if len(recent_data) > 2 else 0
            }
            
            # Convert to DataFrame
            features_df = pd.DataFrame([features])
            
            # Scale features
            features_scaled = self.scaler.transform(features_df)
            
            # Make prediction (with log transform reversal)
            log_prediction = self.model.predict(features_scaled)[0]
            kwh_prediction = np.expm1(log_prediction)
            
            # Add some uncertainty (confidence interval)
            lower_bound = kwh_prediction * 0.85  # 15% lower
            upper_bound = kwh_prediction * 1.15  # 15% higher
            
            # Add to predictions
            predictions.append({
                'prediction_date': pred_date.strftime('%Y-%m-%d'),
                'month': month,
                'predicted_kwh': round(kwh_prediction),
                'lower_bound': round(lower_bound),
                'upper_bound': round(upper_bound),
                'avg_daily_temperature': avg_temp
            })
        
        return pd.DataFrame(predictions)
    
    def _prepare_data(self, data):
        """Prepare data for training/prediction"""
        df = data.copy()
        
        # Ensure date columns are datetime
        date_columns = ['bill_date', 'billing_start_date', 'billing_end_date', 'due_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Sort by date
        df = df.sort_values('bill_date')
        
        # Extract month
        df['month'] = df['bill_date'].dt.month
        
        return df
    
    def _engineer_features(self, df, month_lookahead=1):
        """Create features for training"""
        # Need at least 4 months of data plus the target month
        if len(df) < 4 + month_lookahead:
            return pd.DataFrame()
        
        # Create rolling features
        features = []
        
        for i in range(len(df) - month_lookahead):
            # Target is usage 'month_lookahead' months ahead
            target_idx = i + month_lookahead
            
            feature_row = {
                'month': df.iloc[target_idx]['month'],
                'avg_daily_temperature': df.iloc[target_idx]['avg_daily_temperature'],
                'days_in_billing_period': df.iloc[target_idx]['days_in_billing_period'],
                'avg_3m_kwh': df.iloc[max(0, i-2):i+1]['kwh_used'].mean(),
                'last_month_kwh': df.iloc[i]['kwh_used'],
                'last_2_month_kwh': df.iloc[i-1]['kwh_used'] if i > 0 else 0,
                'last_3_month_kwh': df.iloc[i-2]['kwh_used'] if i > 1 else 0,
                'target_kwh': df.iloc[target_idx]['kwh_used']
            }
            
            features.append(feature_row)
        
        return pd.DataFrame(features)
    
    def _default_temp_for_month(self, month):
        """Return default temperature for a month if no historical data available"""
        if month in [12, 1, 2]:  # Winter
            return 35
        elif month in [3, 4, 5]:  # Spring
            return 55
        elif month in [6, 7, 8]:  # Summer
            return 75
        else:  # Fall
            return 60
    
    def _save_model(self):
        """Save the trained model and scaler"""
        model_path = os.path.join(self.model_dir, 'usage_predictor_model.pkl')
        scaler_path = os.path.join(self.model_dir, 'usage_predictor_scaler.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Model saved to {model_path}")
    
    def _load_model(self):
        """Load the trained model and scaler"""
        model_path = os.path.join(self.model_dir, 'usage_predictor_model.pkl')
        scaler_path = os.path.join(self.model_dir, 'usage_predictor_scaler.pkl')
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            return True
        except FileNotFoundError:
            print(f"Model files not found at {model_path}")
            return False
    def _engineer_enhanced_features(self, df, month_lookahead=1):
        """Create enhanced features for training with cyclical encoding and interactions"""
        # Need at least 4 months of data plus the target month
        if len(df) < 4 + month_lookahead:
            return pd.DataFrame()
        
        # Create rolling features with enhanced engineering
        features = []
        
        for i in range(len(df) - month_lookahead):
            # Target is usage 'month_lookahead' months ahead
            target_idx = i + month_lookahead
            
            # Get date-based features with cyclical encoding
            month = df.iloc[target_idx]['month']
            
            feature_row = {
                'month': month,
                'month_sin': np.sin(2 * np.pi * month / 12),
                'month_cos': np.cos(2 * np.pi * month / 12),
                'avg_daily_temperature': df.iloc[target_idx]['avg_daily_temperature'],
                'days_in_billing_period': df.iloc[target_idx]['days_in_billing_period'],
                # Interaction terms
                'avg_temp_x_month': df.iloc[target_idx]['avg_daily_temperature'] * month,
                'days_x_temp': df.iloc[target_idx]['days_in_billing_period'] * df.iloc[target_idx]['avg_daily_temperature'],
                # Include recent usage history
                'avg_3m_kwh': df.iloc[max(0, i-2):i+1]['kwh_used'].mean(),
                'last_month_kwh': df.iloc[i]['kwh_used'],
                'last_2_month_kwh': df.iloc[i-1]['kwh_used'] if i > 0 else 0,
                'last_3_month_kwh': df.iloc[i-2]['kwh_used'] if i > 1 else 0,
                # Target
                'target_kwh': df.iloc[target_idx]['kwh_used']
            }
            
            features.append(feature_row)
        
        return pd.DataFrame(features)