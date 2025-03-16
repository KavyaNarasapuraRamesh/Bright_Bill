import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

class UsagePredictor:
    def __init__(self):
        """Initialize the usage predictor"""
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        
    def predict(self, historical_data, future_months=3):
        """
        Predict future usage based on historical data
        
        Args:
            historical_data: DataFrame with historical bill data
            future_months: Number of months to predict into the future
            
        Returns:
            DataFrame with predictions or None if insufficient data
        """
        try:
            # Make sure we have at least 2 data points
            if len(historical_data) < 2:
                print("Not enough historical data for accurate predictions")
                # If we have at least one data point, we can use it as a baseline
                if len(historical_data) == 1:
                    return self._generate_simple_prediction(historical_data, future_months)
                return None
            
            # Make sure required columns exist
            required_columns = ['bill_date', 'kwh_used']
            if not all(col in historical_data.columns for col in required_columns):
                print("Missing required columns in historical data")
                return None
            
            # Sort by bill date
            df = historical_data.sort_values('bill_date')
            
            # Extract features
            X, y = self._extract_features(df)
            
            if X.shape[0] < 2:
                print("Not enough valid data points after feature extraction")
                return self._generate_simple_prediction(historical_data, future_months)
            
            # Train model
            self.model.fit(X, y)
            
            # Generate future date features
            last_date = df['bill_date'].max()
            future_dates = [last_date + timedelta(days=30 * (i + 1)) for i in range(future_months)]
            
            # Create future feature dataframe
            future_features = pd.DataFrame({
                'month': [d.month for d in future_dates],
                'year': [d.year for d in future_dates],
                'day_of_year': [d.timetuple().tm_yday for d in future_dates]
            })
            
            # Add more features if available
            if 'avg_daily_temperature' in df.columns:
                # If we have temperature data, use the average or a seasonal approximation
                if df['avg_daily_temperature'].notnull().any():
                    # Use average for same month from historical data
                    for i, date in enumerate(future_dates):
                        month_data = df[df['bill_date'].dt.month == date.month]
                        if not month_data.empty and month_data['avg_daily_temperature'].notnull().any():
                            future_features.loc[i, 'avg_daily_temperature'] = month_data['avg_daily_temperature'].mean()
                        else:
                            # Fallback to a seasonal approximation (higher in summer, lower in winter)
                            month = date.month
                            if month in [12, 1, 2]:  # Winter
                                future_features.loc[i, 'avg_daily_temperature'] = 40
                            elif month in [3, 4, 5]:  # Spring
                                future_features.loc[i, 'avg_daily_temperature'] = 65
                            elif month in [6, 7, 8]:  # Summer
                                future_features.loc[i, 'avg_daily_temperature'] = 85
                            else:  # Fall
                                future_features.loc[i, 'avg_daily_temperature'] = 65
            
            # Scale features if we have enough data
            if X.shape[0] >= 3:
                X_scaled = self.scaler.fit_transform(X)
                future_X_scaled = self.scaler.transform(future_features)
                
                # Predict
                predictions = self.model.predict(future_X_scaled)
            else:
                # For limited data, use simpler prediction
                predictions = self.model.predict(future_features)
            
            # Apply constraints to avoid unrealistic predictions
            avg_usage = df['kwh_used'].mean()
            std_usage = df['kwh_used'].std() if len(df) > 1 else avg_usage * 0.2
            
            # Constrain predictions to be within reasonable bounds (±30% of average)
            min_usage = max(avg_usage * 0.7, 0)  # Don't go below 0
            max_usage = avg_usage * 1.3
            
            # Apply bounds with smoothing for consecutive months
            bounded_predictions = []
            for i, pred in enumerate(predictions):
                if i == 0:
                    # First month can only deviate 20% from most recent bill
                    last_usage = df['kwh_used'].iloc[-1]
                    lower = max(last_usage * 0.8, min_usage)
                    upper = min(last_usage * 1.2, max_usage)
                else:
                    # Subsequent months can deviate 15% from previous prediction
                    last_pred = bounded_predictions[-1]
                    lower = max(last_pred * 0.85, min_usage)
                    upper = min(last_pred * 1.15, max_usage)
                
                bounded_predictions.append(max(min(pred, upper), lower))
            
            # Create result dataframe
            result = pd.DataFrame({
                'prediction_date': future_dates,
                'predicted_kwh': bounded_predictions,
                'lower_bound': [max(p * 0.9, 0) for p in bounded_predictions],
                'upper_bound': [p * 1.1 for p in bounded_predictions],
                'avg_daily_temperature': future_features['avg_daily_temperature'] if 'avg_daily_temperature' in future_features.columns else None
            })
            
            return result
            
        except Exception as e:
            print(f"Error in usage prediction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_features(self, df):
        """Extract features from historical data"""
        # Basic features
        features = pd.DataFrame({
            'month': df['bill_date'].dt.month,
            'year': df['bill_date'].dt.year,
            'day_of_year': df['bill_date'].dt.dayofyear
        })
        
        # Add temperature if available
        if 'avg_daily_temperature' in df.columns and df['avg_daily_temperature'].notnull().any():
            features['avg_daily_temperature'] = df['avg_daily_temperature']
        
        # Target is kWh used
        target = df['kwh_used'].values
        
        return features, target
    
    def _generate_simple_prediction(self, historical_data, future_months):
        """Generate simple predictions when we have limited data"""
        # Get the most recent bill
        df = historical_data.sort_values('bill_date')
        last_bill = df.iloc[-1]
        
        # Use the last kWh value as baseline
        baseline_kwh = last_bill['kwh_used']
        
        # Generate future dates
        last_date = last_bill['bill_date']
        future_dates = [last_date + timedelta(days=30 * (i + 1)) for i in range(future_months)]
        
        # Create predictions with small variations (±5% per month)
        predictions = []
        for i, date in enumerate(future_dates):
            month = date.month
            
            # Seasonal adjustment
            if month in [12, 1, 2]:  # Winter
                adjustment = 1.05  # 5% increase for heating
            elif month in [6, 7, 8]:  # Summer
                adjustment = 1.08  # 8% increase for cooling
            else:
                adjustment = 1.0  # No change in spring/fall
            
            # Apply small trend (max ±5% per month)
            trend = 1.0 + (0.05 * (i + 1) / future_months)
            
            predicted_kwh = baseline_kwh * adjustment * trend
            
            predictions.append({
                'prediction_date': date,
                'predicted_kwh': round(predicted_kwh, 2),
                'lower_bound': round(predicted_kwh * 0.9, 2),
                'upper_bound': round(predicted_kwh * 1.1, 2),
                'avg_daily_temperature': None  # No temperature data for simple predictions
            })
        
        return pd.DataFrame(predictions)