# ml_models/train_and_evaluate_models.py
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

def generate_dummy_bill_data(num_samples=200):
    """Generate dummy electricity bill data for training"""
    data = []
    
    # Set date range for bills
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = (end_date - start_date).days
    
    # Typical household metrics
    household_sizes = [1, 2, 3, 4, 5]
    home_sizes = [800, 1000, 1200, 1500, 1800, 2200, 2500, 3000]
    
    # Temperature varies by month
    monthly_temps = {
        1: 35, 2: 40, 3: 50, 4: 60, 5: 70, 6: 80,
        7: 85, 8: 83, 9: 75, 10: 65, 11: 55, 12: 40
    }
    
    # Electricity rates
    base_rate = 0.12  # dollars per kWh
    
    for _ in range(num_samples):
        # Generate household characteristics
        household_size = np.random.choice(household_sizes)
        home_sqft = np.random.choice(home_sizes)
        
        # Generate random bill date
        days_offset = np.random.randint(0, date_range)
        bill_date = start_date + timedelta(days=days_offset)
        month = bill_date.month
        
        # Temperature affects energy usage
        avg_temp = monthly_temps[month] + np.random.uniform(-5, 5)
        
        # Calculate base usage (influenced by household size, home size, temperature)
        # Winter months: Higher usage if colder
        is_winter = month in [12, 1, 2]
        is_summer = month in [6, 7, 8, 9]
        
        # Base formula: household size + home size + temperature influence
        base_kwh = (household_size * 100) + (home_sqft * 0.1)
        
        # Temperature adjustment
        if is_winter:
            temp_factor = (70 - avg_temp) * 5  # More usage when colder
            base_kwh += temp_factor
        elif is_summer:
            temp_factor = (avg_temp - 70) * 5  # More usage when hotter (AC)
            base_kwh += temp_factor
        
        # Add randomness for natural variation
        kwh_used = base_kwh * np.random.uniform(0.85, 1.15)
        
        # Calculate bill amount
        days_in_period = np.random.randint(28, 32)
        avg_daily_usage = kwh_used / days_in_period
        total_bill = kwh_used * base_rate * np.random.uniform(0.95, 1.05)
        
        # Create bill record
        bill_record = {
            'account_number': f"ACC{100000 + _}",
            'customer_name': f"Customer {_}",
            'bill_date': bill_date.strftime('%Y-%m-%d'),
            'kwh_used': round(kwh_used, 2),
            'days_in_billing_period': days_in_period,
            'avg_daily_usage': round(avg_daily_usage, 2),
            'avg_daily_temperature': round(avg_temp, 1),
            'total_bill_amount': round(total_bill, 2),
            'household_size': household_size,
            'home_sqft': home_sqft
        }
        
        data.append(bill_record)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV for future use
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/bill_training_data.csv', index=False)
    
    print(f"Generated {num_samples} bill records")
    return df

def generate_dummy_appliance_data(num_samples=200):
    """Generate dummy appliance usage data for training"""
    data = []
    
    # Appliance metrics
    appliance_data = {
        'air_conditioner': {'avg_wattage': 1500, 'max_hours': 12},
        'refrigerator': {'avg_wattage': 150, 'max_hours': 24},
        'electric_water_heater': {'avg_wattage': 4000, 'max_hours': 3},
        'clothes_dryer': {'avg_wattage': 3000, 'max_hours': 3},
        'washing_machine': {'avg_wattage': 500, 'max_hours': 2}
    }
    
    household_sizes = [1, 2, 3, 4, 5]
    home_sizes = [800, 1000, 1200, 1500, 1800, 2200, 2500, 3000]
    
    for _ in range(num_samples):
        # Generate household characteristics
        household_size = np.random.choice(household_sizes)
        home_sqft = np.random.choice(home_sizes)
        
        # Generate appliance usage
        appliance_usage = {}
        total_kwh = 0
        
        for appliance, specs in appliance_data.items():
            # Hours depends on household size
            if appliance == 'refrigerator':
                hours = 24  # Always on
            else:
                scale_factor = 1.0
                if appliance == 'air_conditioner':
                    scale_factor = (home_sqft / 1500) * np.random.uniform(0.5, 1.5)
                elif appliance == 'electric_water_heater':
                    scale_factor = (household_size / 3) * np.random.uniform(0.8, 1.2)
                elif appliance in ['clothes_dryer', 'washing_machine']:
                    scale_factor = (household_size / 3) * np.random.uniform(0.7, 1.3)
                
                max_hours = specs['max_hours']
                hours = np.random.uniform(0, max_hours) * scale_factor
                hours = min(hours, max_hours)  # Cap at maximum hours
            
            appliance_usage[f"{appliance}_hours"] = round(hours, 1)
            
            # Calculate monthly kWh: (wattage * hours * 30 days) / 1000
            kwh = (specs['avg_wattage'] * hours * 30) / 1000
            total_kwh += kwh
        
        # Add some base usage for other appliances not explicitly modeled
        base_usage = (household_size * 50) + (home_sqft * 0.05)
        total_kwh += base_usage
        
        # Add randomness
        total_kwh *= np.random.uniform(0.9, 1.1)
        
        # Create record
        record = {
            'household_size': household_size,
            'home_sqft': home_sqft,
            'total_kwh': round(total_kwh, 2)
        }
        record.update(appliance_usage)
        
        data.append(record)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV for future use
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/appliance_training_data.csv', index=False)
    
    print(f"Generated {num_samples} appliance usage records")
    return df

def generate_combined_data(num_samples=200):
    """Generate dummy data that combines bill and appliance usage"""
    # Generate basic bill data first
    bill_df = generate_dummy_bill_data(num_samples)
    
    # Generate appliance usage based on the bill data
    data = []
    
    appliance_data = {
        'air_conditioner': {'avg_wattage': 1500, 'max_hours': 12},
        'refrigerator': {'avg_wattage': 150, 'max_hours': 24},
        'electric_water_heater': {'avg_wattage': 4000, 'max_hours': 3},
        'clothes_dryer': {'avg_wattage': 3000, 'max_hours': 3},
        'washing_machine': {'avg_wattage': 500, 'max_hours': 2}
    }
    
    for _, bill in bill_df.iterrows():
        # Start with bill data
        record = bill.to_dict()
        
        # Convert bill date to datetime for calculations
        bill_date = datetime.strptime(record['bill_date'], '%Y-%m-%d')
        month = bill_date.month
        
        # Determine seasonal factors
        is_winter = month in [12, 1, 2]
        is_summer = month in [6, 7, 8, 9]
        
        # Total energy from bill
        total_kwh = record['kwh_used']
        
        # Calculate appliance usage that would approximately sum to this kWh
        household_size = record['household_size']
        home_sqft = record['home_sqft']
        
        # Base usage for miscellaneous appliances
        base_usage = (household_size * 50) + (home_sqft * 0.05)
        remaining_kwh = total_kwh - base_usage
        
        # Distribute remaining energy among appliances
        appliance_usage = {}
        
        # Refrigerator is always on
        fridge_kwh = (appliance_data['refrigerator']['avg_wattage'] * 24 * 30) / 1000
        remaining_kwh -= fridge_kwh
        appliance_usage['refrigerator_hours'] = 24
        
        # Distribute remaining kWh proportionally with seasonal adjustments
        if is_winter:
            # More heating, less AC in winter
            weights = {
                'air_conditioner': 0.1,
                'electric_water_heater': 0.5,
                'clothes_dryer': 0.2,
                'washing_machine': 0.2
            }
        elif is_summer:
            # More AC, less heating in summer
            weights = {
                'air_conditioner': 0.6,
                'electric_water_heater': 0.2,
                'clothes_dryer': 0.1,
                'washing_machine': 0.1
            }
        else:
            # Balanced in spring/fall
            weights = {
                'air_conditioner': 0.3,
                'electric_water_heater': 0.3,
                'clothes_dryer': 0.2,
                'washing_machine': 0.2
            }
        
        # Apply randomness to weights
        for appliance in weights:
            weights[appliance] *= np.random.uniform(0.7, 1.3)
        
        # Normalize weights
        total_weight = sum(weights.values())
        for appliance in weights:
            weights[appliance] /= total_weight
        
        # Calculate appliance kWh and hours
        for appliance, weight in weights.items():
            appliance_kwh = remaining_kwh * weight
            specs = appliance_data[appliance]
            
            # Calculate hours: (kWh * 1000) / (wattage * 30 days)
            hours = (appliance_kwh * 1000) / (specs['avg_wattage'] * 30)
            
            # Cap at maximum hours
            hours = min(hours, specs['max_hours'])
            
            # Add randomness
            hours *= np.random.uniform(0.9, 1.1)
            
            appliance_usage[f"{appliance}_hours"] = round(hours, 1)
        
        # Add appliance usage to record
        record.update(appliance_usage)
        
        data.append(record)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV for future use
    df.to_csv('data/processed/combined_training_data.csv', index=False)
    
    print(f"Generated {len(df)} combined bill+appliance records")
    return df

def train_bill_prediction_model():
    """Train and save the bill prediction model"""
    print("Training bill prediction model...")
    
    # Load or generate data
    data_path = 'data/processed/bill_training_data.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        df = generate_dummy_bill_data(200)
    
    # Process dates
    df['bill_date'] = pd.to_datetime(df['bill_date'])
    df['month'] = df['bill_date'].dt.month
    
    # Features and target
    features = ['household_size', 'home_sqft', 'avg_daily_temperature', 'month']
    target = 'kwh_used'
    
    X = df[features]
    y = df[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Bill Model Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    
    # Feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    
    with open('models/bill_predictor_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/bill_predictor_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Bill prediction model saved to 'models/bill_predictor_model.pkl'")
    
    # Visualize actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual kWh')
    plt.ylabel('Predicted kWh')
    plt.title('Bill Model: Actual vs Predicted')
    
    # Save figure
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/bill_model_evaluation.png')
    plt.close()
    
    return model, scaler

def train_appliance_prediction_model():
    """Train and save the appliance usage prediction model"""
    print("Training appliance prediction model...")
    
    # Load or generate data
    data_path = 'data/processed/appliance_training_data.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        df = generate_dummy_appliance_data(200)
    
    # Features and target
    appliance_cols = [col for col in df.columns if col.endswith('_hours')]
    features = appliance_cols + ['household_size', 'home_sqft']
    target = 'total_kwh'
    
    X = df[features]
    y = df[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Appliance Model Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    
    # Feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Save model and scaler
    with open('models/appliance_predictor_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/appliance_predictor_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Appliance prediction model saved to 'models/appliance_predictor_model.pkl'")
    
    # Visualize actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual kWh')
    plt.ylabel('Predicted kWh')
    plt.title('Appliance Model: Actual vs Predicted')
    
    # Save figure
    plt.savefig('outputs/appliance_model_evaluation.png')
    plt.close()
    
    return model, scaler

def train_combined_prediction_model():
    """Train and save the combined bill+appliance prediction model"""
    print("Training combined prediction model...")
    
    # Load or generate data
    data_path = 'data/processed/combined_training_data.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        df = generate_combined_data(200)
    
    # Process dates
    df['bill_date'] = pd.to_datetime(df['bill_date'])
    df['month'] = df['bill_date'].dt.month
    
    # Features and target
    bill_features = ['household_size', 'home_sqft', 'avg_daily_temperature', 'month']
    appliance_cols = [col for col in df.columns if col.endswith('_hours')]
    features = bill_features + appliance_cols
    
    # Use total_kwh as the target variable, not kwh_used
    target = 'total_kwh'  # CHANGED FROM 'kwh_used' TO 'total_kwh'
    
    X = df[features]
    y = df[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Combined Model Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    
    # Feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Save model and scaler
    with open('models/combined_predictor_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/combined_predictor_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Combined prediction model saved to 'models/combined_predictor_model.pkl'")
    
    # Visualize actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual kWh')
    plt.ylabel('Predicted kWh')
    plt.title('Combined Model: Actual vs Predicted')
    
    # Save figure
    plt.savefig('outputs/combined_model_evaluation.png')
    plt.close()
    
    return model, scaler

def evaluate_model_with_real_data(model_type='combined', sample_data=None):
    """Evaluate a trained model with real or sample data"""
    print(f"Evaluating {model_type} model with sample data...")
    
    if sample_data is None:
        # Use a simple test case
        if model_type == 'bill':
            sample_data = {
                'household_size': 3,
                'home_sqft': 1800,
                'avg_daily_temperature': 75,
                'month': 7  # July
            }
        elif model_type == 'appliance':
            sample_data = {
                'air_conditioner_hours': 8.0,
                'refrigerator_hours': 24.0,
                'electric_water_heater_hours': 1.5,
                'clothes_dryer_hours': 1.0,
                'washing_machine_hours': 0.5,
                'household_size': 3,
                'home_sqft': 1800
            }
        else:  # combined
            sample_data = {
                'household_size': 3,
                'home_sqft': 1800,
                'avg_daily_temperature': 75,
                'month': 7,
                'air_conditioner_hours': 8.0,
                'refrigerator_hours': 24.0,
                'electric_water_heater_hours': 1.5,
                'clothes_dryer_hours': 1.0,
                'washing_machine_hours': 0.5
            }
    
    # Load model and scaler
    model_path = f'models/{model_type}_predictor_model.pkl'
    scaler_path = f'models/{model_type}_predictor_scaler.pkl'
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        print(f"Model files not found. Please train the {model_type} model first.")
        return None
    
    # Convert sample to DataFrame
    X = pd.DataFrame([sample_data])
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    
    print(f"Sample input: {sample_data}")
    print(f"Predicted kWh: {prediction:.2f}")
    
    # Calculate cost (using average rate of $0.15 per kWh)
    estimated_cost = prediction * 0.15
    print(f"Estimated monthly cost: ${estimated_cost:.2f}")
    
    return prediction

if __name__ == "__main__":
    print("Starting model training and evaluation...")
    
    # Create directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # Generate data if needed
    if not os.path.exists('data/processed/bill_training_data.csv'):
        bill_data = generate_dummy_bill_data(200)
    
    if not os.path.exists('data/processed/appliance_training_data.csv'):
        appliance_data = generate_dummy_appliance_data(200)
    
    if not os.path.exists('data/processed/combined_training_data.csv'):
        combined_data = generate_combined_data(200)
    
    # Train models
    bill_model, bill_scaler = train_bill_prediction_model()
    appliance_model, appliance_scaler = train_appliance_prediction_model()
    combined_model, combined_scaler = train_combined_prediction_model()
    
    # Evaluate with sample data
    print("\n=== Model Evaluation with Sample Data ===")
    evaluate_model_with_real_data('bill')
    evaluate_model_with_real_data('appliance')
    evaluate_model_with_real_data('combined')
    
    print("\nTraining and evaluation complete!")
    print("Model files saved to 'models/' directory")
    print("Evaluation figures saved to 'outputs/' directory")