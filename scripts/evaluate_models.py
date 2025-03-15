import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models.usage_predictor import UsagePredictor
from ml_models.cost_predictor import CostPredictor

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

def evaluate_usage_predictor(data, k_folds=3):
    """
    Evaluate the usage predictor with improved features and model
    
    Args:
        data: DataFrame with bill data
        k_folds: Number of folds for cross-validation
    """
    print("\n=== Evaluating Usage Predictor ===")
    
    # Sort data by date
    df = data.copy().sort_values('bill_date')
    
    # We need at least k_folds + 1 data points
    if len(df) < k_folds + 1:
        print(f"Not enough data for {k_folds}-fold validation. Need at least {k_folds+1} bills.")
        return
    
    # Prepare metrics collections
    all_actuals = []
    all_predictions = []
    
    # Create output directory for evaluation
    os.makedirs('data/processed/evaluation', exist_ok=True)
    
    # For low data scenarios, use leave-one-out validation
    for i in range(len(df) - 1):
        # Use all bills except the next one for training
        train_data = df.drop(df.index[i+1])
        test_bill = df.iloc[i+1]
        
        try:
            # Enhanced feature engineering
            train_features = pd.DataFrame({
                'month': train_data['bill_date'].dt.month,
                'month_sin': np.sin(2 * np.pi * train_data['bill_date'].dt.month / 12),
                'month_cos': np.cos(2 * np.pi * train_data['bill_date'].dt.month / 12),
                'avg_daily_temperature': train_data['avg_daily_temperature'],
                'days_in_billing_period': train_data['days_in_billing_period'],
                'avg_temp_x_month': train_data['avg_daily_temperature'] * train_data['bill_date'].dt.month,
                'days_x_temp': train_data['days_in_billing_period'] * train_data['avg_daily_temperature']
            })
            
            # Create and fit scaler
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            train_features_scaled = scaler.fit_transform(train_features)
            
            # Log-transform the target for better handling of high variance
            train_target = np.log1p(train_data['kwh_used'])
            
            # Try XGBoost instead of RandomForest
            from xgboost import XGBRegressor
            model = XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
            model.fit(train_features_scaled, train_target)
            
            # Create test features with same transformations
            test_features = pd.DataFrame([{
                'month': test_bill['bill_date'].month,
                'month_sin': np.sin(2 * np.pi * test_bill['bill_date'].month / 12),
                'month_cos': np.cos(2 * np.pi * test_bill['bill_date'].month / 12),
                'avg_daily_temperature': test_bill['avg_daily_temperature'],
                'days_in_billing_period': test_bill['days_in_billing_period'],
                'avg_temp_x_month': test_bill['avg_daily_temperature'] * test_bill['bill_date'].month,
                'days_x_temp': test_bill['days_in_billing_period'] * test_bill['avg_daily_temperature']
            }])
            
            # Scale test features
            test_features_scaled = scaler.transform(test_features)
            
            # Make prediction and reverse log transform
            log_prediction = model.predict(test_features_scaled)[0]
            prediction = np.expm1(log_prediction)
            
            # Store actual and predicted values
            actual = float(test_bill['kwh_used'])
            all_actuals.append(actual)
            all_predictions.append(float(prediction))
            
            print(f"Fold {i+1}: Actual: {actual}, Predicted: {round(prediction)}, Error: {round(abs(actual - prediction))} kWh")
        
        except Exception as e:
            print(f"Error in fold {i+1}: {e}")
            continue
    
    if not all_actuals or not all_predictions:
        print("No valid predictions were generated. Check your data and model.")
        return None
    
    # Calculate metrics
    mae = float(mean_absolute_error(all_actuals, all_predictions))
    rmse = float(np.sqrt(mean_squared_error(all_actuals, all_predictions)))
    mape = float(np.mean(np.abs((np.array(all_actuals) - np.array(all_predictions)) / np.array(all_actuals))) * 100)
    r2 = float(r2_score(all_actuals, all_predictions))
    
    print("\nUsage Prediction Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.2f} kWh")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} kWh")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"R-squared (R²): {r2:.4f}")
    
    # Save metrics
    metrics = {
        'model': 'Usage Predictor',
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'actuals': [float(x) for x in all_actuals],
        'predictions': [float(x) for x in all_predictions]
    }
    
    with open('data/processed/evaluation/usage_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.plot(all_actuals, label='Actual Usage', marker='o')
    plt.plot(all_predictions, label='Predicted Usage', marker='x')
    plt.xlabel('Bill Number')
    plt.ylabel('kWh Usage')
    plt.title('Usage Prediction vs Actual')
    plt.legend()
    plt.grid(True)
    plt.savefig('data/processed/evaluation/usage_prediction_plot.png')
    
    return metrics

def evaluate_cost_predictor(data, usage_metrics):
    """
    Evaluate the cost predictor with improved methods
    
    Args:
        data: DataFrame with bill data
        usage_metrics: Metrics from usage prediction evaluation
    """
    print("\n=== Evaluating Cost Predictor ===")
    
    # Train cost predictor on all data
    cost_predictor = CostPredictor()
    cost_predictor.train(data)
    
    # Get actual and predicted usage values
    actuals = usage_metrics['actuals']
    predicted_usage = usage_metrics['predictions']
    
    # For each pair, calculate cost and compare to actual
    all_actual_costs = []
    all_predicted_costs = []
    
    for i, (actual_usage, pred_usage) in enumerate(zip(actuals, predicted_usage)):
        # Get the bill with this actual usage
        bill_index = len(data) - len(actuals) + i
        if bill_index < len(data):
            actual_bill = data.iloc[bill_index]
            
            # Get actual total cost
            actual_cost = float(actual_bill['total_bill_amount'])
            all_actual_costs.append(actual_cost)
            
            # Predict cost based on predicted usage
            cost_prediction = cost_predictor.predict_cost(pred_usage)
            if cost_prediction:
                predicted_cost = float(cost_prediction['total_bill_amount'])
                all_predicted_costs.append(predicted_cost)
                
                print(f"Bill {i+1}: Actual: ${actual_cost:.2f}, Predicted: ${predicted_cost:.2f}, Error: ${abs(actual_cost - predicted_cost):.2f}")
    
    # Calculate metrics
    mae = float(mean_absolute_error(all_actual_costs, all_predicted_costs))
    rmse = float(np.sqrt(mean_squared_error(all_actual_costs, all_predicted_costs)))
    mape = float(np.mean(np.abs((np.array(all_actual_costs) - np.array(all_predicted_costs)) / np.array(all_actual_costs))) * 100)
    r2 = float(r2_score(all_actual_costs, all_predicted_costs))
    
    print("\nCost Prediction Metrics:")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"R-squared (R²): {r2:.4f}")
    
    # Save metrics
    metrics = {
        'model': 'Cost Predictor',
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'actuals': [float(x) for x in all_actual_costs],
        'predictions': [float(x) for x in all_predicted_costs]
    }
    
    with open('data/processed/evaluation/cost_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.plot(all_actual_costs, label='Actual Cost', marker='o')
    plt.plot(all_predicted_costs, label='Predicted Cost', marker='x')
    plt.xlabel('Bill Number')
    plt.ylabel('Bill Amount ($)')
    plt.title('Cost Prediction vs Actual')
    plt.legend()
    plt.grid(True)
    plt.savefig('data/processed/evaluation/cost_prediction_plot.png')
    
    return metrics

def main():
    """Evaluate all prediction models"""
    print("Evaluating prediction models...")
    
    # Load data
    data = load_data()
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    print(f"Loaded {len(data)} bills for evaluation")
    
    # Evaluate usage predictor
    usage_metrics = evaluate_usage_predictor(data)
    
    # Evaluate cost predictor
    if usage_metrics:
        cost_metrics = evaluate_cost_predictor(data, usage_metrics)
    
    print("\nModel evaluation completed!")

if __name__ == "__main__":
    main()