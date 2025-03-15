from services.extraction_service import BillExtractionService

def main():
    # Create extraction service
    extractor = BillExtractionService(
        raw_folder='data/raw',
        processed_folder='data/processed'
    )
    
    # Process all bills and save to CSV
    bills_df = extractor.process_all_bills()
    
    if bills_df is not None:
        # Print summary of extracted data
        print("\nExtracted Bills Summary:")
        print(f"Total bills: {len(bills_df)}")
        print(f"Date range: {bills_df['billing_start_date'].min()} to {bills_df['billing_end_date'].max()}")
        print(f"Total kWh: {bills_df['kwh_used'].sum()}")
        print(f"Total amount: ${bills_df['total_bill_amount'].sum():.2f}")
        
        # Show preview of the data
        print("\nData Preview:")
        print(bills_df[['bill_date', 'kwh_used', 'total_bill_amount']].head())

if __name__ == "__main__":
    main()