import os
import json
import fitz  # PyMuPDF
import google.generativeai as genai
from PIL import Image
import re

from dotenv import load_dotenv

load_dotenv()

def extract_bill_data(pdf_path, api_key):
    """Extract comprehensive data from a bill PDF using Gemini API"""
    try:
        # Set up Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Render PDF to image
        doc = fitz.open(pdf_path)
        page = doc[0]
        zoom = 2.0  # Higher resolution
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Save as temporary PNG
        temp_image = "temp_bill.png"
        pix.save(temp_image)
        
        # Open with PIL to ensure it's valid
        img = Image.open(temp_image)
        
        # Prepare prompt for extraction - requesting all fields needed for the SQL schema
        prompt = """
            Extract these details from this electricity bill as JSON:
            - account_number: Account number
            - customer_name: Customer name
            - billing_start_date: Start date of billing period
            - billing_end_date: End date of billing period
            - days_in_billing_period: Number of days in billing period
            - bill_date: Date when bill was issued
            - due_date: Payment due date
            - kwh_used: Total kWh consumed
            - meter_start_value: Starting meter reading
            - meter_end_value: Ending meter reading
            - avg_daily_usage: Average daily usage in kWh
            - avg_daily_temperature: Average daily temperature
            - total_bill_amount: Total amount due
            - utility_price_to_compare: Utility price to compare (in cents per kWh)
            - supplier_rate: Look for "Commodity Charge: X Kh § Y" where Y is the supplier rate (in dollars per kWh)
            - customer_charge: Customer charge amount
            - distribution_related_component: Distribution related component
            - cost_recovery_charges: Cost recovery charges
            - consumer_rate_credit: Consumer rate credit
            - distribution_credit: Distribution credit (if applicable)
            - non_standard_credit: Non-standard credit (if applicable)
            - utility_charges: Total utility charges
            - supplier_charges: Total supplier charges
                    
        Return ONLY valid JSON with these fields.
        """
        
        # Send to Gemini
        response = model.generate_content([prompt, img])
        
        # Clean up temp file
        os.remove(temp_image)
        
        # Parse the response
        text = response.text
        print(f"Processing {os.path.basename(pdf_path)}:")
        
        # Extract JSON
        json_pattern = r'({[\s\S]*})'
        match = re.search(json_pattern, text)
        
        if match:
            json_str = match.group(1)
            # Try to parse as JSON
            try:
                data = json.loads(json_str)
                
                # Save to JSON file
                output_dir = 'data/processed'
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(pdf_path))[0]}.json")
                
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"Successfully extracted data and saved to {output_file}")
                return data
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Attempted to parse: {json_str[:100]}...")
                return None
        else:
            print("No JSON data found in response")
            return None
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def process_all_bills(raw_folder, api_key):
    """Process all bills in the folder"""
    results = []
    
    for filename in os.listdir(raw_folder):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(raw_folder, filename)
            data = extract_bill_data(pdf_path, api_key)
            if data:
                results.append(data)
    
    # Save combined results
    if results:
        with open('data/processed/all_bills.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved combined data for {len(results)} bills to data/processed/all_bills.json")
    
    return results

def extract_historical_usage(pdf_path, api_key):
    """Separate function to extract historical usage from each bill"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Render PDF to image
        doc = fitz.open(pdf_path)
        page = doc[0]
        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        temp_image = "temp_bill_hist.png"
        pix.save(temp_image)
        
        img = Image.open(temp_image)
        
        # Prompt specifically for historical usage
        prompt = """
        Extract the historical usage data from this electricity bill. 
        Look for the 'Usage History' section that contains monthly usage data.
        
        Return only a JSON array of objects, with each object containing:
        - month: The month (e.g., "Dec 23", "Jan 24", etc.)
        - kwh: The kilowatt-hour usage for that month (as a number)
        
        Example format:
        [
          {"month": "Dec 23", "kwh": 1502},
          {"month": "Jan 24", "kwh": 1807}
        ]
        """
        
        response = model.generate_content([prompt, img])
        os.remove(temp_image)
        
        text = response.text
        
        # Extract JSON array
        json_pattern = r'(\[[\s\S]*\])'
        match = re.search(json_pattern, text)
        
        if match:
            json_str = match.group(1)
            try:
                data = json.loads(json_str)
                
                # Save historical data
                output_dir = 'data/processed'
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_history.json")
                
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"Successfully extracted historical data and saved to {output_file}")
                return data
            except json.JSONDecodeError as e:
                print(f"JSON parsing error for historical data: {e}")
                return None
        else:
            print("No historical usage data found")
            return None
            
    except Exception as e:
        print(f"Error extracting historical data: {e}")
        return None

if __name__ == "__main__":
    api_key = os.getenv("GEMINI_API_KEY")
    raw_folder = 'data/raw'
    
    # Process all bills
    print("=== Extracting Main Bill Data ===")
    bills = process_all_bills(raw_folder, api_key)
    
    # Process historical usage separately
    print("\n=== Extracting Historical Usage Data ===")
    historical_data = {}
    for filename in os.listdir(raw_folder):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(raw_folder, filename)
            history = extract_historical_usage(pdf_path, api_key)
            if history:
                historical_data[filename] = history
    
    # Save combined historical data
    if historical_data:
        with open('data/processed/historical_usage.json', 'w') as f:
            json.dump(historical_data, f, indent=2)
        print(f"Saved historical usage data for {len(historical_data)} bills")
    
    # Print summary
    if bills:
        print("\n=== Extraction Summary ===")
        print(f"Total bills processed: {len(bills)}")
        
        # Print sample data from first bill
        print("\nSample data from first bill:")
        first_bill = bills[0]
        for key, value in first_bill.items():
            print(f"{key}: {value}")