import numpy as np
import pandas as pd
import pickle
import os

class CostPredictor:
    def __init__(self, model_dir='models'):
        """Initialize the cost predictor model"""
        self.model_dir = model_dir
        self.rates = {}
        self.charge_ratios = {}
        os.makedirs(model_dir, exist_ok=True)
    
    def train(self, data):
        """
        Train the cost predictor model
        
        Args:
            data: DataFrame containing historical bill data
        """
        print("Training cost prediction model...")
        
        # Extract the supplier rate (should be consistent)
        self.rates['supplier_rate'] = data['supplier_rate'].mean()
        self.rates['utility_price_to_compare'] = data['utility_price_to_compare'].mean()
        
        # Calculate ratios of different charges to kwh_used
        charge_types = [
            'distribution_related_component',
            'cost_recovery_charges'
        ]
        
        for charge in charge_types:
            if charge in data.columns:
                # Calculate ratio to kwh_used
                self.charge_ratios[charge] = data[charge].sum() / data['kwh_used'].sum()
        
        # Save model
        self._save_model()
        
        return True
    
    def predict_cost(self, kwh_prediction):
        """
        Predict the cost for a given kwh prediction
        
        Args:
            kwh_prediction: Predicted kwh usage
            
        Returns:
            Dictionary with cost components
        """
        if not self.rates:
            self._load_model()
            if not self.rates:
                print("No trained model found. Please train the model first.")
                return None
        
        # Fixed charges
        customer_charge = 4.00
        consumer_rate_credit = -1.02
        
        # Calculate variable charges
        distribution_related_component = kwh_prediction * self.charge_ratios.get('distribution_related_component', 0.05)
        cost_recovery_charges = kwh_prediction * self.charge_ratios.get('cost_recovery_charges', 0.03)
        
        # Winter credits - assume none for prediction
        distribution_credit = 0
        non_standard_credit = 0
        
        # Calculate totals
        utility_charges = round(customer_charge + distribution_related_component + 
                            cost_recovery_charges + consumer_rate_credit + 
                            distribution_credit + non_standard_credit, 2)
        
        supplier_charges = round(kwh_prediction * self.rates.get('supplier_rate', 0.119), 2)
        total_bill_amount = round(utility_charges + supplier_charges, 2)
        
        return {
            'customer_charge': customer_charge,
            'distribution_related_component': round(distribution_related_component, 2),
            'cost_recovery_charges': round(cost_recovery_charges, 2),
            'consumer_rate_credit': consumer_rate_credit,
            'distribution_credit': distribution_credit,
            'non_standard_credit': non_standard_credit,
            'utility_charges': utility_charges,
            'supplier_charges': supplier_charges,
            'total_bill_amount': total_bill_amount
        }
    
    def _save_model(self):
        """Save the trained ratios and rates"""
        model_path = os.path.join(self.model_dir, 'cost_predictor_data.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump({'rates': self.rates, 'charge_ratios': self.charge_ratios}, f)
        
        print(f"Cost model saved to {model_path}")
    
    def _load_model(self):
        """Load the trained ratios and rates"""
        model_path = os.path.join(self.model_dir, 'cost_predictor_data.pkl')
        
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.rates = data.get('rates', {})
                self.charge_ratios = data.get('charge_ratios', {})
            
            return True
        except FileNotFoundError:
            print(f"Cost model file not found at {model_path}")
            return False