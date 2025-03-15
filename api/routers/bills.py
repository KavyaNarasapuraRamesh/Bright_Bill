from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from typing import List
import pandas as pd
import json
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.models.schemas import BillBase, BillCreate, BillResponse

router = APIRouter()

# Load data helper function
def load_bill_data():
    try:
        with open('data/processed/combined_bills.json', 'r') as f:
            return json.load(f)
    except:
        return []

@router.get("/", response_model=List[BillResponse])
async def get_all_bills():
    """
    Get all electricity bills
    """
    bills = load_bill_data()
    for i, bill in enumerate(bills):
        bill['id'] = i + 1
    return bills

@router.get("/{bill_id}", response_model=BillResponse)
async def get_bill(bill_id: int):
    """
    Get a specific bill by ID
    """
    bills = load_bill_data()
    if bill_id < 1 or bill_id > len(bills):
        raise HTTPException(status_code=404, detail="Bill not found")
    
    bill = bills[bill_id - 1]
    bill['id'] = bill_id
    return bill

@router.post("/", response_model=BillResponse)
async def create_bill(bill: BillCreate):
    """
    Add a new bill
    """
    bills = load_bill_data()
    
    # Convert to dict for storage
    new_bill = bill.dict()
    bills.append(new_bill)
    
    # Save back to file
    with open('data/processed/combined_bills.json', 'w') as f:
        json.dump(bills, f, indent=2, default=str)
    
    # Return with ID
    new_bill['id'] = len(bills)
    return new_bill

@router.post("/upload", response_model=BillResponse)
async def upload_bill_pdf(file: UploadFile = File(...)):
    """
    Upload a bill PDF for processing
    """
    # Save uploaded PDF
    filename = f"data/raw/{file.filename}"
    with open(filename, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Here you would call your extraction service
    # For now, just return a placeholder
    return {
        "id": 0,
        "account_number": "placeholder",
        "bill_date": "2025-01-01",
        "billing_start_date": "2025-01-01", 
        "billing_end_date": "2025-01-31",
        "days_in_billing_period": 31,
        "kwh_used": 0,
        "meter_start_value": 0,
        "meter_end_value": 0,
        "avg_daily_usage": 0,
        "avg_daily_temperature": 0,
        "total_bill_amount": 0,
        "utility_price_to_compare": 0,
        "supplier_rate": 0,
        "customer_charge": 0,
        "distribution_related_component": 0,
        "cost_recovery_charges": 0,
        "consumer_rate_credit": 0,
        "distribution_credit": 0,
        "non_standard_credit": 0,
        "utility_charges": 0,
        "supplier_charges": 0
    }