from fastapi import APIRouter, HTTPException, Depends
from typing import List
import sys
import os
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.models.schemas import AnomalyResponse
from services.prediction_service import PredictionService

router = APIRouter()

# Dependency to get prediction service
def get_prediction_service():
    return PredictionService()

@router.get("/{bill_id}", response_model=List[AnomalyResponse])
async def detect_anomalies(
    bill_id: int,
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """
    Detect anomalies in a specific bill
    """
    # Load bill data
    try:
        with open('data/processed/combined_bills.json', 'r') as f:
            bills = json.load(f)
    except:
        bills = []
    
    if bill_id < 1 or bill_id > len(bills):
        raise HTTPException(status_code=404, detail="Bill not found")
    
    # Get the bill
    bill = bills[bill_id - 1]
    
    # Detect anomalies
    anomalies = prediction_service.detect_bill_anomalies(bill)
    
    # Format the response
    response = []
    for anomaly in anomalies:
        response.append({
            'bill_id': bill_id,
            **anomaly
        })
    
    return response