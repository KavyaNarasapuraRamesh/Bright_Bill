# retrain_models.py
import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import json
import traceback

# Make sure all required directories exist
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("Starting model retraining process...")

# Check if we have any existing bill data
combined_path = 'data/processed/combined_bills.json'
try:
    if os.path.exists(combined_path):
        with open(combined_path, 'r') as f:
            bill_data = json.load(f)
        
        if bill_data:
            print(f"Found {len(bill_data)} existing bills for training")
        else:
            print("No existing bill data found. Using sample data for initial models.")
            bill_data = []
    else:
        print("No combined_bills.json file found. Creating empty file.")
        with open(combined_path, 'w') as f:
            json.dump([], f)
        bill_data = []
except Exception as e:
    print(f"Error loading bill data: {e}")
    bill_data = []

# Generate sample data if we don't have real data
if not bill_data:
    print("Generating sample bill data for initial models")
    
    # Create 12 months of sample data
    sample_data = []
    base_usage = 600  # Base usage in kWh
    
    for month in range(1, 13):
        # Seasonal adjustments (more moderate)
        if month in [12, 1, 2]:  # Winter
            seasonal_factor = 1.15
        elif month in [6, 7, 8]:  # Summer
            seasonal_factor = 1.2
        else:
            seasonal_factor = 1.0
        
        # Add some randomness (but more constrained)
        random_factor = np.random.uniform(0.95, 1.05)
        
        usage = base_usage * seasonal_factor * random_factor
        bill_amount = usage * 0.15  # $0.15 per kWh
        
        # Create bill data
        bill = {
            "account_number": "SAMPLE-123",
            "customer_name": "Sample Customer",
            "bill_date": f"2024-{month:02d}-15",
            "billing_start_date": f"2024-{month:02d}-01",
            "billing_end_date": f"2024-{month:02d}-30",
            "due_date": f"2024-{month+1 if month < 12 else 1:02d}-15",
            "kwh_used": round(usage, 2),
            "total_bill_amount": round(bill_amount, 2),
            "utility_charges": round(bill_amount * 0.4, 2),
            "supplier_charges": round(bill_amount * 0.6, 2),
            "avg_daily_usage": round(usage / 30, 2),
            "avg_daily_temperature": round(70 + (month - 1) * 3 if month <= 6 else 70 + (12 - month) * 3, 1)
        }
        
        sample_data.append(bill)
    
    # Use the sample data for training
    bill_data = sample_data
    
    # Save sample data to JSON file
    with open(combined_path, 'w') as f:
        json.dump(sample_data, f, indent=2)

# Convert to DataFrame for easier processing
df = pd.DataFrame(bill_data)

# Convert date columns if present
date_columns = ['bill_date', 'billing_start_date', 'billing_end_date', 'due_date']
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# 1. Bill Prediction Model
print("\n[1/4] Training bill prediction model...")
try:
    # Extract features for bill prediction
    if 'bill_date' in df.columns and 'kwh_used' in df.columns:
        # Create feature DataFrame
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
        
        # Train a simple model - LinearRegression is more conservative than tree-based models
        model = LinearRegression()
        scaler = StandardScaler()
        
        if len(df) >= 3:
            X_scaled = scaler.fit_transform(features)
            model.fit(X_scaled, target)
        else:
            model.fit(features, target)
        
        # Save the model and scaler
        with open('models/bill_predictor_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        with open('models/bill_predictor_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        print("✓ Bill predictor model trained and saved successfully")
    else:
        print("✗ Missing required columns for bill predictor model")
except Exception as e:
    print(f"✗ Error generating bill predictor model: {e}")
    traceback.print_exc()

# 2. Appliance Prediction Model
print("\n[2/4] Training appliance predictor model...")
try:
    # Create a simple model for appliance prediction
    # Sample data with more conservative scaling
    appliance_features = np.array([
        [0, 24, 0, 0, 0, 2, 1200],  # air_conditioner, refrigerator, water_heater, clothes_dryer, washing_machine, household_size, home_sqft
        [4, 24, 1, 0.5, 0.5, 3, 1800],
        [8, 24, 2, 1, 1, 4, 2400],
        [12, 24, 3, 1.5, 1.5, 5, 3000]
    ])
    
    # More moderate target values
    appliance_target = np.array([400, 650, 900, 1200])  # Corresponding kWh usage
    
    # Create and train the model with LinearRegression
    appliance_model = LinearRegression()
    appliance_scaler = StandardScaler()
    
    X_scaled = appliance_scaler.fit_transform(appliance_features)
    appliance_model.fit(X_scaled, appliance_target)
    
    # Save the model and scaler
    with open('models/appliance_predictor_model.pkl', 'wb') as f:
        pickle.dump(appliance_model, f)
    
    with open('models/appliance_predictor_scaler.pkl', 'wb') as f:
        pickle.dump(appliance_scaler, f)
    
    print("✓ Appliance predictor model trained and saved successfully")
except Exception as e:
    print(f"✗ Error generating appliance predictor model: {e}")
    traceback.print_exc()

# 3. Combined Predictor Model
print("\n[3/4] Training combined predictor model...")
try:
    # Create sample data with more moderate predictions
    combined_features = np.array([
        [2, 1200, 65, 1, 0, 24, 0, 0, 0],  # household_size, home_sqft, avg_temp, month, air_con, fridge, water, dryer, washer
        [3, 1800, 75, 6, 4, 24, 1, 0.5, 0.5],
        [4, 2400, 85, 7, 8, 24, 2, 1, 1],
        [5, 3000, 75, 10, 2, 24, 3, 1.5, 1.5]
    ])
    
    # More moderate target values
    combined_target = np.array([450, 800, 950, 780])  # Corresponding kWh usage
    
    # Create and train the model
    combined_model = LinearRegression()
    combined_scaler = StandardScaler()
    
    X_scaled = combined_scaler.fit_transform(combined_features)
    combined_model.fit(X_scaled, combined_target)
    
    # Save the model and scaler
    with open('models/combined_predictor_model.pkl', 'wb') as f:
        pickle.dump(combined_model, f)
    
    with open('models/combined_predictor_scaler.pkl', 'wb') as f:
        pickle.dump(combined_scaler, f)
    
    print("✓ Combined predictor model trained and saved successfully")
except Exception as e:
    print(f"✗ Error generating combined predictor model: {e}")
    traceback.print_exc()

# 4. Anomaly Detector
print("\n[4/4] Training anomaly detector...")
try:
    # Calculate basic statistics for z-score based detection
    thresholds = {}
    for col in ['kwh_used', 'avg_daily_usage', 'total_bill_amount']:
        if col in df.columns:
            thresholds[col + '_mean'] = df[col].mean()
            thresholds[col + '_std'] = df[col].std() if len(df) > 1 else df[col].mean() * 0.2
    
    # Save thresholds
    with open('models/anomaly_detector_thresholds.pkl', 'wb') as f:
        pickle.dump(thresholds, f)
    
    # We'll skip training the Isolation Forest for now
    print("✓ Anomaly detector thresholds saved successfully")
except Exception as e:
    print(f"✗ Error training anomaly detector: {e}")
    traceback.print_exc()

print("\nModel retraining complete! New model files have been created in the 'models' directory.")
print("\nTest predictions with these models:")

# Try a test prediction with the new bill model
try:
    # Load the model
    with open('models/bill_predictor_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('models/bill_predictor_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Test input
    test_input = {
        'month': 7,
        'year': 2024,
        'day_of_year': 186
    }
    
    if 'avg_daily_temperature' in features.columns:
        test_input['avg_daily_temperature'] = 75
    
    # Convert to DataFrame and scale
    X_test = pd.DataFrame([test_input])
    X_test_scaled = scaler.transform(X_test)
    
    # Predict
    prediction = model.predict(X_test_scaled)[0]
    
    print(f"\nTest bill prediction (July): {prediction:.2f} kWh")
    print(f"Estimated bill amount: ${prediction * 0.15:.2f}")
except Exception as e:
    print(f"Error testing bill model: {e}")

print("\nYou can now start your FastAPI application with the newly trained models.")