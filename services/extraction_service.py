import os
import re
import pandas as pd
from datetime import datetime
import pytesseract
from pdf2image import convert_from_path
from utils.pdf_utils import preprocess_image
from utils.date_utils import parse_date

# Set the Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\pbhatta12\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

class BillExtractionService:
    def __init__(self, raw_folder='data/raw', processed_folder='data/processed'):
        self.raw_folder = raw_folder
        self.processed_folder = processed_folder
        self.poppler_path = r'C:\Users\pbhatta12\OneDrive - University of Toledo\Desktop\Hackathoon\Release-24.08.0-0\poppler-24.08.0\Library\bin'
        os.makedirs(self.processed_folder, exist_ok=True)
        
    def extract_text_from_pdf(self, pdf_path):
        """Convert PDF to text using OCR"""
        try:
            # Convert PDF to images using the explicit poppler path
            images = convert_from_path(pdf_path, poppler_path=self.poppler_path)
            
            # Extract text from each page
            full_text = ""
            for img in images:
                # Optional: preprocess image to improve OCR accuracy
                processed_img = preprocess_image(img)
                
                # Perform OCR
                text = pytesseract.image_to_string(processed_img)
                full_text += text + "\n"
                
            return full_text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return None
    
    def parse_bill_data(self, text):
        """Extract structured data from bill text using regex patterns"""
        data = {}
        
        # Account information
        account_match = re.search(r"Account Number:?\s*(\d{3}\s*\d{3}\s*\d{3}\s*\d{3})", text)
        if account_match:
            data['account_number'] = account_match.group(1).replace(" ", "")
        
        # Customer name
        name_match = re.search(r"Bill For:.*?\n(.*?)\n", text)
        if name_match:
            data['customer_name'] = name_match.group(1).strip()
        
        # Billing period
        period_match = re.search(r"Billing Period:.*?(\w+\s+\d+)\s+to\s+(\w+\s+\d+),\s+(\d{4})\s+for\s+(\d+)\s+days", text)
        if period_match:
            start_date_str = f"{period_match.group(1)} {period_match.group(3)}"
            end_date_str = f"{period_match.group(2)} {period_match.group(3)}"
            data['billing_start_date'] = parse_date(start_date_str)
            data['billing_end_date'] = parse_date(end_date_str)
            data['days_in_billing_period'] = int(period_match.group(4))
        
        # Bill date and due date
        bill_date_match = re.search(r"(\w+\s+\d+,\s+\d{4})\nBill Based On", text)
        if bill_date_match:
            data['bill_date'] = parse_date(bill_date_match.group(1))
            
        due_date_match = re.search(r"Due Date:?\s*(\w+\s+\d+,\s+\d{4})", text)
        if due_date_match:
            data['due_date'] = parse_date(due_date_match.group(1))
        
        # Usage information
        usage_match = re.search(r"KWH used\s+(\d+)", text)
        if usage_match:
            data['kwh_used'] = int(usage_match.group(1))
        
        # Meter readings
        meter_start_match = re.search(r"KWH Reading \(Actual\)\s+(\d+,\d+)", text, re.MULTILINE)
        meter_end_match = re.search(r"KWH Reading \(Actual\)\s+(\d+,\d+)", text)
        if meter_start_match and meter_end_match:
            # Get the second occurrence for start (previous reading)
            start_readings = re.findall(r"KWH Reading \(Actual\)\s+(\d+,\d+)", text)
            if len(start_readings) >= 2:
                data['meter_start_value'] = int(start_readings[1].replace(",", ""))
                data['meter_end_value'] = int(meter_end_match.group(1).replace(",", ""))
        
        # Average daily usage and temperature
        avg_usage_match = re.search(r"Average Daily Use \(KWH\).*?(\d+)", text)
        if avg_usage_match:
            data['avg_daily_usage'] = int(avg_usage_match.group(1))
            
        avg_temp_match = re.search(r"Average Daily Temperature.*?(\d+)", text)
        if avg_temp_match:
            data['avg_daily_temperature'] = int(avg_temp_match.group(1))
        
        # Bill amount
        amount_match = re.search(r"Amount Due:?\s*\$(\d+\.\d+)", text)
        if amount_match:
            data['total_bill_amount'] = float(amount_match.group(1))
        
        # Utility price to compare
        ptc_match = re.search(r"Residential Service.*?(\d+\.\d+)\s+cents per KWH", text)
        if ptc_match:
            data['utility_price_to_compare'] = float(ptc_match.group(1)) / 100  # Convert cents to dollars
        
        # Supplier rate
        supplier_rate_match = re.search(r"Commodity Charge:.*?(\d+)\s+Kh\s+§\s+(\d+\.\d+)", text)
        if supplier_rate_match:
            data['supplier_rate'] = float(supplier_rate_match.group(2))
        
        # Charges breakdown
        customer_charge_match = re.search(r"Customer Charge\s+(\d+\.\d+)", text)
        if customer_charge_match:
            data['customer_charge'] = float(customer_charge_match.group(1))
            
        distribution_match = re.search(r"Distribution Related Component\s+(\d+\.\d+)", text)
        if distribution_match:
            data['distribution_related_component'] = float(distribution_match.group(1))
            
        recovery_match = re.search(r"Cost Recovery Charges\s+(\d+\.\d+)", text)
        if recovery_match:
            data['cost_recovery_charges'] = float(recovery_match.group(1))
            
        credit_match = re.search(r"Consumer Rate Credit\s+(-\d+\.\d+)", text)
        if credit_match:
            data['consumer_rate_credit'] = float(credit_match.group(1))
            
        # Distribution credit (winter only)
        dist_credit_match = re.search(r"Residential Distribution Credit\s+(-\d+\.\d+)", text)
        if dist_credit_match:
            data['distribution_credit'] = float(dist_credit_match.group(1))
        else:
            data['distribution_credit'] = 0.0
            
        # Non-standard credit (winter only)
        non_std_credit_match = re.search(r"Residential Non-Standard Credit\s+(-\d+\.\d+)", text)
        if non_std_credit_match:
            data['non_standard_credit'] = float(non_std_credit_match.group(1))
        else:
            data['non_standard_credit'] = 0.0
            
        # Total charges
        utility_charges_match = re.search(r"Total Charges\s+\$\s+(\d+\.\d+)", text)
        if utility_charges_match:
            data['utility_charges'] = float(utility_charges_match.group(1))
            
        supplier_charges_match = re.search(r"Total Alpha Gas & Electric, LLC Current Charges\s+(\d+\.\d+)", text)
        if supplier_charges_match:
            data['supplier_charges'] = float(supplier_charges_match.group(1))
        
        return data
    
    def process_bill(self, pdf_filename):
        """Process a single bill PDF and extract data"""
        pdf_path = os.path.join(self.raw_folder, pdf_filename)
        
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return None
        
        # Parse bill data
        bill_data = self.parse_bill_data(text)
        
        # Validate extracted data
        if not self._validate_bill_data(bill_data):
            print(f"Validation failed for bill: {pdf_filename}")
            return None
        
        return bill_data
    
    def process_all_bills(self):
        """Process all bills in the raw folder and save to CSV"""
        all_bills = []
        
        # Process each PDF in the raw folder
        for filename in os.listdir(self.raw_folder):
            if filename.lower().endswith('.pdf'):
                print(f"Processing {filename}...")
                bill_data = self.process_bill(filename)
                if bill_data:
                    all_bills.append(bill_data)
        
        # Convert to DataFrame and save to CSV
        if all_bills:
            df = pd.DataFrame(all_bills)
            output_path = os.path.join(self.processed_folder, 'electricity_bills.csv')
            df.to_csv(output_path, index=False)
            print(f"Saved {len(all_bills)} bills to {output_path}")
            return df
        else:
            print("No bills were successfully processed.")
            return None
    
    def _validate_bill_data(self, data):
        """Basic validation of extracted bill data"""
        required_fields = [
            'account_number', 'bill_date', 'billing_start_date', 
            'billing_end_date', 'kwh_used', 'total_bill_amount'
        ]
        
        # Check that all required fields are present
        for field in required_fields:
            if field not in data or data[field] is None:
                print(f"Missing required field: {field}")
                return False
                
        # Check that numeric fields are reasonable
        if data.get('kwh_used', 0) <= 0:
            print("Invalid kWh usage")
            return False
            
        if data.get('total_bill_amount', 0) <= 0:
            print("Invalid bill amount")
            return False
        
        return True