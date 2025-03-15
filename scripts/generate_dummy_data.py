import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta

def generate_appliance_data(num_samples=100):
    """Generate dummy data for appliance usage model training"""
    np.random.seed(42)  # For reproducibility
    
    # Create empty DataFrame
    data = []
    
    # Generate data for different household profiles
    household_sizes = [1, 2, 3, 4, 5]
    home_sizes = [800, 1200, 1500, 1800, 2200, 2800, 3500]
    
    # Base usage patterns for different months (seasonal effects)
    seasonal_factors = {
        1: 1.2,   # January (winter)
        2: 1.15,  # February
        3: 1.0,   # March
        4: 0.9,   # April
        5: 0.85,  # May
        6: 1.1,   # June (summer starts)
        7: 1.3,   # July (peak summer)
        8: 1.25,  # August
        9: 0.95,  # September
        10: 0.9,  # October
        11: 1.0,  # November
        12: 1.1   # December (winter starts)
    }
    
    # Generate samples
    for _ in range(num_samples):
        # Household characteristics
        household_size = np.random.choice(household_sizes)
        home_sqft = np.random.choice(home_sizes)
        month = np.random.randint(1, 13)  # 1-12
        seasonal_factor = seasonal_factors[month]
        
        # Appliance usage (hours per day)
        ac_hours = np.random.uniform(0, 12) * (1.5 if month in [6, 7, 8, 9] else 0.5)
        refrigerator_hours = 24  # Always on
        water_heater_hours = np.random.uniform(0.5, 3) * (1.2 if month in [1, 2, 12] else 0.9)
        dryer_hours = np.random.uniform(0, 3) * (household_size / 3)
        washer_hours = np.random.uniform(0, 2) * (household_size / 3)
        
        # Calculate base energy for each appliance (monthly kWh)
        # Formula: (wattage * hours per day * 30 days) / 1000
        ac_energy = (1500 * ac_hours * 30) / 1000
        refrigerator_energy = (150 * refrigerator_hours * 30) / 1000
        water_heater_energy = (4000 * water_heater_hours * 30) / 1000
        dryer_energy = (3000 * dryer_hours * 30) / 1000
        washer_energy = (500 * washer_hours * 30) / 1000
        
        # Other energy usage that varies by home size and occupants
        other_energy = (home_sqft * 0.05) + (household_size * 30)
        
        # Calculate total energy with seasonal adjustments and some randomness
        base_total = (ac_energy + refrigerator_energy + water_heater_energy + 
                     dryer_energy + washer_energy + other_energy)
        
        total_energy = base_total * seasonal_factor * np.random.uniform(0.9, 1.1)
        
        # Add to dataset
        sample = {
            'month': month,
            'household_size': household_size,
            'home_sqft': home_sqft,
            'air_conditioner_hours': ac_hours,
            'refrigerator_hours': refrigerator_hours,
            'electric_water_heater_hours': water_heater_hours,
            'clothes_dryer_hours': dryer_hours,
            'washing_machine_hours': washer_hours,
            'total_kwh': total_energy
        }
        
        data.append(sample)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save the dataset
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/appliance_training_data.csv', index=False)
    
    print(f"Generated {num_samples} appliance usage samples")
    return df

def generate_combined_data(num_samples=100):
    """Generate dummy data that combines bill data with appliance usage"""
    np.random.seed(42)  # For reproducibility
    
    # First generate appliance data
    appliance_df = generate_appliance_data(num_samples)
    
    # Create additional bill-specific fields
    data = []
    
    start_date = datetime(2023, 1, 1)
    
    for idx, row in appliance_df.iterrows():
        # Generate a bill date
        bill_date = start_date + timedelta(days=np.random.randint(0, 365))
        
        # Bill specific data
        bill_record = {
            'account_number': f"ACC{100000 + idx}",
            'customer_name': f"Customer {idx}",
            'bill_date': bill_date.strftime('%Y-%m-%d'),
            'billing_period_days': np.random.randint(28, 33),
            'avg_daily_usage': row['total_kwh'] / 30,
            'avg_daily_temperature': 70 + (10 if row['month'] in [6, 7, 8] else -10 if row['month'] in [12, 1, 2] else 0),
            'supplier_rate': np.random.uniform(0.11, 0.14),
            
            # Copy appliance data
            'household_size': row['household_size'],
            'home_sqft': row['home_sqft'],
            'air_conditioner_hours': row['air_conditioner_hours'],
            'refrigerator_hours': row['refrigerator_hours'],
            'electric_water_heater_hours': row['electric_water_heater_hours'],
            'clothes_dryer_hours': row['clothes_dryer_hours'],
            'washing_machine_hours': row['washing_machine_hours'],
            
            # Energy and cost data
            'total_kwh': row['total_kwh'],
            'total_bill_amount': row['total_kwh'] * np.random.uniform(0.14, 0.17)
        }
        
        data.append(bill_record)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save the dataset
    df.to_csv('data/processed/combined_training_data.csv', index=False)
    
    print(f"Generated {num_samples} combined bill+appliance samples")
    return df

if __name__ == "__main__":
    print("Generating dummy training data...")
    
    # Generate both datasets
    appliance_data = generate_appliance_data(100)
    combined_data = generate_combined_data(100)
    
    print("Data generation complete!")
    print(f"Appliance data shape: {appliance_data.shape}")
    print(f"Combined data shape: {combined_data.shape}")