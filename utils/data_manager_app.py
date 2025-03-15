import os
import json
import pandas as pd
from datetime import datetime

def save_bill_data_to_history(bill_data):
    """Save bill data to a historical dataset for improving predictions"""
    try:
        # Path for historical data
        history_path = 'data/processed/historical_bills.json'
        
        # Load existing historical data or create a new dataset
        try:
            with open(history_path, 'r') as f:
                historical_bills = json.load(f)
                print(f"Loaded {len(historical_bills)} historical bills")
        except (FileNotFoundError, json.JSONDecodeError):
            historical_bills = []
            print("Creating new historical bills dataset")
        
        # Check if we already have this bill in our history (avoid duplicates)
        is_duplicate = False
        for existing_bill in historical_bills:
            # Compare key fields to determine if this is a duplicate
            if (existing_bill.get('account_number') == bill_data.get('account_number') and
                existing_bill.get('bill_date') == bill_data.get('bill_date') and
                existing_bill.get('kwh_used') == bill_data.get('kwh_used')):
                is_duplicate = True
                break
        
        # Only add if it's not a duplicate
        if not is_duplicate:
            historical_bills.append(bill_data)
            
            # Save updated history
            with open(history_path, 'w') as f:
                json.dump(historical_bills, f, indent=2, default=str)
            print(f"Added new bill to historical dataset (total: {len(historical_bills)})")
            
            # Periodically retrain models if we have enough new data
            if len(historical_bills) % 5 == 0:  # Retrain after every 5 new bills
                print("Dataset has grown - scheduling model retraining")
                # This could be a background task for a real app
                # For hackathon, we can use a simple flag file:
                os.makedirs('models', exist_ok=True)
                with open('models/retrain_needed.txt', 'w') as f:
                    f.write(str(datetime.now()))
        else:
            print("Bill already exists in historical dataset - skipping")
                
        return True
    except Exception as e:
        print(f"Error saving bill to history: {str(e)}")
        return False

def save_appliance_data(account_number, appliance_data):
    """Save appliance usage data linked to an account"""
    try:
        # Path for appliance data
        appliance_path = 'data/processed/appliance_usage.json'
        
        # Load existing data or create new
        try:
            with open(appliance_path, 'r') as f:
                usage_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            usage_data = {}
        
        # Store data by account number
        usage_data[account_number] = {
            'timestamp': datetime.now().isoformat(),
            'data': appliance_data
        }
        
        # Save updated data
        with open(appliance_path, 'w') as f:
            json.dump(usage_data, f, indent=2, default=str)
        
        return True
    except Exception as e:
        print(f"Error saving appliance data: {str(e)}")
        return False

def get_appliance_data(account_number):
    """Retrieve appliance usage data for an account"""
    try:
        appliance_path = 'data/processed/appliance_usage.json'
        
        try:
            with open(appliance_path, 'r') as f:
                usage_data = json.load(f)
                
            if account_number in usage_data:
                return usage_data[account_number]['data']
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        
        return None
    except Exception as e:
        print(f"Error retrieving appliance data: {str(e)}")
        return None

def combine_bill_with_appliance_data(bill_data, appliance_data):
    """Combine bill and appliance data for enhanced prediction"""
    combined = {}
    
    # Copy bill data
    for key, value in bill_data.items():
        combined[key] = value
    
    # Add appliance data with prefixes
    for appliance, hours in appliance_data.items():
        combined[f'appliance_{appliance}_hours'] = hours
    
    return combined