import os
import json
import pandas as pd
from datetime import datetime
from pdf2image import convert_from_path
import google.generativeai as genai
from PIL import Image
import tempfile

class GeminiBillExtractionService:
    def __init__(self, api_key, raw_folder='data/raw', processed_folder='data/processed'):
        self.raw_folder = raw_folder
        self.processed_folder = processed_folder
        self.gemini_model = None
        self.setup_gemini(api_key)
        os.makedirs(self.processed_folder, exist_ok=True)
    
    def setup_gemini(self, api_key):
        """Set up the Gemini API with the provided key"""
        try:
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            print("Gemini model initialized successfully")
        except Exception as e:
            print(f"Error initializing Gemini model: {e}")
            self.gemini_model = None
    
    def extract_data_from_bill(self, pdf_path):
        """Extract data from bill using Gemini model"""
        if self.gemini_model is None:
            print("Gemini model not initialized")
            return None
            
        try:
            # Convert PDF to image
            images = convert_from_path(pdf_path, dpi=300)
            
            # Save first page as temp image
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                temp_image_path = tmp.name
                images[0].save(temp_image_path, "JPEG")
            
            # Open image for Gemini
            image = Image.open(temp_image_path)
            
            # Prepare prompt for Gemini
            prompt = """
            Extract the following key details from this electricity bill in JSON format:
            1. account_number: Account number
            2. customer_name: Customer name
            3. billing_start_date: Start date of billing period (YYYY-MM-DD)
            4. billing_end_date: End date of billing period (YYYY-MM-DD)
            5. days_in_billing_period: Number of days in billing period
            6. bill_date: Date when bill was issued (YYYY-MM-DD)
            7. due_date: Payment due date (YYYY-MM-DD)
            8. kwh_used: Total kWh consumed
            9. meter_start_value: Starting meter reading
            10. meter_end_value: Ending meter reading
            11. avg_daily_usage: Average daily usage in kWh
            12. avg_daily_temperature: Average daily temperature
            13. total_bill_amount: Total amount due
            14. utility_price_to_compare: Utility price to compare (in $ per kWh)
            15. supplier_rate: Supplier rate (in $ per kWh)
            16. customer_charge: Customer charge
            17. distribution_related_component: Distribution related component
            18. cost_recovery_charges: Cost recovery charges
            19. consumer_rate_credit: Consumer rate credit
            20. distribution_credit: Distribution credit (if applicable)
            21. non_standard_credit: Non-standard credit (if applicable)
            22. utility_charges: Total utility charges
            23. supplier_charges: Total supplier charges
            24. historical_usage: Array of historical usage data (monthly kWh for last 12 months if available)

            Format dates as YYYY-MM-DD and convert all numeric values to appropriate types.
            Return the output as valid JSON.
            """
            
            # Run Gemini model
            response = self.gemini_model.generate_content([prompt, image])
            
            # Parse the response
            response_text = response.text
            
            # Clean up response text to extract valid JSON
            if "```json" in response_text:
                # Extract JSON between code blocks
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            else:
                # Try to find JSON in the response
                json_str = response_text
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Clean up temporary file
            os.unlink(temp_image_path)
            
            return data
            
        except Exception as e:
            print(f"Error extracting data from bill: {e}")
            return None
    
    def process_bill(self, pdf_filename):
        """Process a single bill PDF and extract data"""
        pdf_path = os.path.join(self.raw_folder, pdf_filename)
        
        print(f"Processing {pdf_filename}...")
        bill_data = self.extract_data_from_bill(pdf_path)
        
        if bill_data:
            # Check if we got historical data as array or convert from object if needed
            if 'historical_usage' in bill_data and isinstance(bill_data['historical_usage'], dict):
                # Convert dict to list format if needed
                bill_data['historical_usage'] = [
                    {"month": k, "usage": v} for k, v in bill_data['historical_usage'].items()
                ]
            
            return bill_data
        else:
            print(f"Failed to extract data from {pdf_filename}")
            return None
    
    def process_all_bills(self):
        """Process all bills in the raw folder and save to CSV"""
        all_bills = []
        historical_data = []
        
        # Process each PDF in the raw folder
        for filename in os.listdir(self.raw_folder):
            if filename.lower().endswith('.pdf'):
                bill_data = self.process_bill(filename)
                if bill_data:
                    # Handle historical usage separately
                    if 'historical_usage' in bill_data and bill_data['historical_usage']:
                        for entry in bill_data['historical_usage']:
                            if isinstance(entry, dict) and 'month' in entry and 'usage' in entry:
                                historical_data.append({
                                    'account_number': bill_data.get('account_number'),
                                    'bill_date': bill_data.get('bill_date'),
                                    'month': entry['month'],
                                    'usage': entry['usage']
                                })
                    
                    # Remove historical_usage from main data
                    if 'historical_usage' in bill_data:
                        del bill_data['historical_usage']
                    
                    all_bills.append(bill_data)
        
        # Convert to DataFrame and save to CSV
        if all_bills:
            df = pd.DataFrame(all_bills)
            output_path = os.path.join(self.processed_folder, 'electricity_bills.csv')
            df.to_csv(output_path, index=False)
            print(f"Saved {len(all_bills)} bills to {output_path}")
            
            # Save historical data if available
            if historical_data:
                hist_df = pd.DataFrame(historical_data)
                hist_output_path = os.path.join(self.processed_folder, 'historical_usage.csv')
                hist_df.to_csv(hist_output_path, index=False)
                print(f"Saved {len(historical_data)} historical usage records to {hist_output_path}")
            
            return df
        else:
            print("No bills were successfully processed.")
            return None