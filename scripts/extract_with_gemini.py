import os
import sys
import dotenv

# Add parent directory to path to import from project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables (optional)
dotenv.load_dotenv()

from services.gemini_extraction_service import GeminiBillExtractionService

def main():
    # Get API key from environment or specify directly
    api_key = os.getenv("GEMINI_API_KEY", "AIzaSyD7sb6xFqv4lWvVW9KhtHVSbJ9tG4L6E2E")
    
    # Create extraction service
    extractor = GeminiBillExtractionService(
        api_key=api_key,
        raw_folder='data/raw',
        processed_folder='data/processed'
    )
    
    # Process all bills and save to CSV
    bills_df = extractor.process_all_bills()
    
    if bills_df is not None:
        # Print summary of extracted data
        print("\nExtracted Bills Summary:")
        print(f"Total bills: {len(bills_df)}")
        
        # Print date range if dates are available
        if 'billing_start_date' in bills_df.columns and 'billing_end_date' in bills_df.columns:
            print(f"Date range: {bills_df['billing_start_date'].min()} to {bills_df['billing_end_date'].max()}")
        
        # Print usage summary if available
        if 'kwh_used' in bills_df.columns:
            print(f"Total kWh: {bills_df['kwh_used'].sum()}")
        
        # Print bill amount summary if available
        if 'total_bill_amount' in bills_df.columns:
            print(f"Total amount: ${bills_df['total_bill_amount'].sum():.2f}")
        
        # Show preview of the data
        print("\nData Preview:")
        preview_cols = [col for col in ['bill_date', 'kwh_used', 'total_bill_amount'] if col in bills_df.columns]
        print(bills_df[preview_cols].head())

if __name__ == "__main__":
    main()