import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class EnsembleSyntheticBillGenerator:
    def __init__(self, real_data_path='data/processed/all_bills.json', 
                 output_dir='data/processed', model_dir='models'):
        """Initialize the synthetic bill generator using ensemble methods"""
        self.real_data_path = real_data_path
        self.output_dir = output_dir
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        
        # Create directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
    
    def load_real_data(self):
        """Load and preprocess real bill data"""
        print("Loading real bill data...")
        try:
            with open(self.real_data_path, 'r') as f:
                real_bills = json.load(f)
            
            # Convert to DataFrame
            df = pd.DataFrame(real_bills)
            
            # Handle date columns
            date_columns = ['bill_date', 'billing_start_date', 'billing_end_date', 'due_date']
            for col in date_columns:
                if col in df.columns:
                    # Try different date formats
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except:
                        pass
            
            # Fill NA values appropriately
            for col in df.columns:
                if df[col].dtype == 'float64' or df[col].dtype == 'int64':
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna('')
            
            # Add month feature for seasonality
            try:
                if 'bill_date' in df.columns:
                    df['month'] = df['bill_date'].dt.month
            except:
                print("Warning: Could not extract month from bill_date")
            
            print(f"Loaded {len(df)} real bills with {len(df.columns)} features")
            return df
        
        except Exception as e:
            print(f"Error loading real data: {e}")
            return pd.DataFrame()
    
    def generate_synthetic_data(self, df, num_samples=12):
        """Generate synthetic bill data based on patterns in real data"""
        print(f"Generating {num_samples} synthetic bills...")
        
        # Create empty dataframe for synthetic data
        synthetic_df = pd.DataFrame()
        
        # Get base information from real data
        account_number = df['account_number'].iloc[0] if 'account_number' in df.columns else "Unknown"
        customer_name = df['customer_name'].iloc[0] if 'customer_name' in df.columns else "Unknown"
        
        # Generate dates first
        bill_dates = self._generate_synthetic_dates(df, num_samples)
        synthetic_df['bill_date'] = bill_dates
        
        # Extract month for seasonality
        synthetic_df['month'] = pd.DatetimeIndex(synthetic_df['bill_date']).month
        
        # Add days in billing period (28-32 days)
        synthetic_df['days_in_billing_period'] = np.random.randint(28, 33, size=num_samples)
        
        # Generate billing_start_date and billing_end_date
        synthetic_df['billing_end_date'] = synthetic_df['bill_date'] - pd.Timedelta(days=1)
        synthetic_df['billing_start_date'] = synthetic_df.apply(
            lambda row: row['billing_end_date'] - pd.Timedelta(days=row['days_in_billing_period']), 
            axis=1
        )
        
        # Generate due_date (20-25 days after bill_date)
        synthetic_df['due_date'] = synthetic_df['bill_date'] + pd.Timedelta(days=20)
        
        # Add account info
        synthetic_df['account_number'] = account_number
        synthetic_df['customer_name'] = customer_name
        
        # Generate temperature based on month
        synthetic_df['avg_daily_temperature'] = synthetic_df['month'].apply(self._temperature_for_month)
        
        # Generate usage based on month and temperature
        synthetic_df['kwh_used'] = synthetic_df.apply(
            lambda row: self._generate_usage(row['month'], row['avg_daily_temperature']), 
            axis=1
        )
        
        # Calculate avg_daily_usage
        synthetic_df['avg_daily_usage'] = (synthetic_df['kwh_used'] / synthetic_df['days_in_billing_period']).round(1)
        
        # Generate rates
        synthetic_df['utility_price_to_compare'] = np.random.uniform(8.0, 9.5, size=len(synthetic_df)).round(2)
        synthetic_df['supplier_rate'] = 0.119  # Fixed based on your data
        
        # Generate charges
        synthetic_df['customer_charge'] = 4.00  # Fixed
        synthetic_df['distribution_related_component'] = (synthetic_df['kwh_used'] * 0.05).round(2)
        synthetic_df['cost_recovery_charges'] = (synthetic_df['kwh_used'] * 0.03).round(2)
        synthetic_df['consumer_rate_credit'] = -1.02  # Fixed
        
        # Winter credits
        synthetic_df['distribution_credit'] = 0.0
        synthetic_df['non_standard_credit'] = 0.0
        
        # Apply winter credits for winter months and usage > 500
        winter_mask = synthetic_df['month'].isin([12, 1, 2]) & (synthetic_df['kwh_used'] > 500)
        synthetic_df.loc[winter_mask, 'distribution_credit'] = (-(synthetic_df.loc[winter_mask, 'kwh_used'] - 500) * 0.003).round(2)
        synthetic_df.loc[winter_mask, 'non_standard_credit'] = (-(synthetic_df.loc[winter_mask, 'kwh_used'] - 500) * 0.003).round(2)
        
        # Calculate total charges
        synthetic_df['utility_charges'] = (
            synthetic_df['customer_charge'] + 
            synthetic_df['distribution_related_component'] + 
            synthetic_df['cost_recovery_charges'] + 
            synthetic_df['consumer_rate_credit'] + 
            synthetic_df['distribution_credit'] + 
            synthetic_df['non_standard_credit']
        ).round(2)
        
        synthetic_df['supplier_charges'] = (synthetic_df['kwh_used'] * synthetic_df['supplier_rate']).round(2)
        synthetic_df['total_bill_amount'] = (synthetic_df['utility_charges'] + synthetic_df['supplier_charges']).round(2)
        
        # Generate meter readings
        synthetic_df['meter_start_value'] = np.random.randint(10000, 20000, size=len(synthetic_df))
        
        # Make meter readings sequential
        for i in range(1, len(synthetic_df)):
            synthetic_df.iloc[i, synthetic_df.columns.get_loc('meter_start_value')] = (
                synthetic_df.iloc[i-1]['meter_start_value'] + 
                synthetic_df.iloc[i-1]['kwh_used']
            )
        
        synthetic_df['meter_end_value'] = synthetic_df['meter_start_value'] + synthetic_df['kwh_used']
        
        # Save synthetic data
        synthetic_path = os.path.join(self.output_dir, 'synthetic_bills.csv')
        synthetic_df.to_csv(synthetic_path, index=False)
        print(f"Saved {len(synthetic_df)} synthetic bills to {synthetic_path}")
        
        # Also save as JSON
        synthetic_json_path = os.path.join(self.output_dir, 'synthetic_bills.json')
        synthetic_df.to_json(synthetic_json_path, orient='records', date_format='iso')
        print(f"Saved synthetic bills to {synthetic_json_path}")
        
        return synthetic_df
    
    def _generate_synthetic_dates(self, df, num_samples):
        """Generate realistic bill dates extending from real data"""
        if 'bill_date' in df.columns:
            dates = pd.to_datetime(df['bill_date'])
            min_date = dates.min()
            max_date = dates.max()
        else:
            # If no dates in real data, use reasonable defaults
            max_date = datetime.now()
            min_date = max_date - timedelta(days=180)
        
        # Generate dates before and after the real data
        before_count = num_samples // 2
        after_count = num_samples - before_count
        
        # Dates before real data
        before_dates = [min_date - timedelta(days=30 * (i + 1)) for i in range(before_count)]
        
        # Dates after real data
        after_dates = [max_date + timedelta(days=30 * (i + 1)) for i in range(after_count)]
        
        # Combine and sort
        all_dates = sorted(before_dates + after_dates)
        
        return pd.to_datetime(all_dates)
    
    def _temperature_for_month(self, month):
        """Generate realistic temperature for a given month"""
        if month in [12, 1, 2]:  # Winter
            return np.random.randint(28, 40)
        elif month in [3, 4, 5]:  # Spring
            return np.random.randint(50, 65)
        elif month in [6, 7, 8]:  # Summer
            return np.random.randint(70, 85)
        else:  # Fall
            return np.random.randint(50, 65)
    
    def _generate_usage(self, month, temperature):
        """Generate realistic usage based on month and temperature with improved seasonality"""
        # Enhanced seasonal base pattern with stronger winter effect
        if month in [12, 1, 2]:  # Winter
            # Much higher winter base usage
            base_usage = np.random.randint(1400, 2000)
            # Stronger temperature effect in winter
            temp_factor = 1.0 + ((40 - temperature) / 80)
        elif month in [3, 4, 5]:  # Spring
            base_usage = np.random.randint(800, 1200)
            temp_factor = 1.0 + ((65 - temperature) / 200)
        elif month in [6, 7, 8]:  # Summer
            base_usage = np.random.randint(400, 700)
            # Stronger cooling effect in summer
            temp_factor = 1.0 + ((temperature - 70) / 150) 
        else:  # Fall
            base_usage = np.random.randint(500, 900)
            temp_factor = 1.0 + ((60 - temperature) / 200)
        
        # Apply temperature factor and small random variation
        # Use Gaussian noise for more realistic variation
        usage = int(base_usage * temp_factor * np.random.normal(1.0, 0.05))
        
        return usage
    
    def combine_real_and_synthetic(self, real_df, synthetic_df):
        """Combine real and synthetic data for a complete dataset"""
        print("Combining real and synthetic data...")
        
        # Ensure compatible columns
        common_columns = list(set(real_df.columns) & set(synthetic_df.columns))
        
        combined_df = pd.concat([
            real_df[common_columns], 
            synthetic_df[common_columns]
        ], ignore_index=True)
        
        # Sort by date
        if 'bill_date' in combined_df.columns:
            combined_df = combined_df.sort_values('bill_date')
        
        # Save combined data
        combined_path = os.path.join(self.output_dir, 'combined_bills.csv')
        combined_df.to_csv(combined_path, index=False)
        
        # Also save as JSON for broader compatibility
        combined_json_path = os.path.join(self.output_dir, 'combined_bills.json')
        combined_df.to_json(combined_json_path, orient='records', date_format='iso')
        
        print(f"Saved {len(combined_df)} combined bills to {combined_path} and {combined_json_path}")
        
        return combined_df

def main():
    # Initialize generator
    generator = EnsembleSyntheticBillGenerator()
    
    # Load and preprocess real data
    real_df = generator.load_real_data()
    if real_df.empty:
        print("No real data available. Exiting.")
        return
    
    # Generate a larger set of synthetic data (100 bills)
    num_synthetic = 100
    print(f"Generating {num_synthetic} synthetic bills...")
    synthetic_df = generator.generate_synthetic_data(real_df, num_synthetic)
    
    # Combine real and synthetic data
    combined_df = generator.combine_real_and_synthetic(real_df, synthetic_df)
    
    # Print summary statistics
    print("\nData Generation Summary:")
    print(f"Real Bills: {len(real_df)}")
    print(f"Synthetic Bills: {len(synthetic_df)}")
    print(f"Combined Bills: {len(combined_df)}")
    
    # Print sample of synthetic data
    print("\nSample of Synthetic Data:")
    sample_cols = ['bill_date', 'kwh_used', 'total_bill_amount']
    sample_cols = [col for col in sample_cols if col in synthetic_df.columns]
    print(synthetic_df[sample_cols].head())

if __name__ == "__main__":
    main()