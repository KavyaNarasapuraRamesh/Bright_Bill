from fastapi import APIRouter, HTTPException, Depends
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.models.schemas import PredictionRequest, PredictionResponse
from services.prediction_service import PredictionService

router = APIRouter()

# Dependency to get prediction service
def get_prediction_service():
    return PredictionService()

@router.post("/", response_model=PredictionResponse)
async def predict_future_bills(
    request: PredictionRequest,
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """
    Predict future bills for an account
    """
    predictions = prediction_service.predict_future_bills(
        request.account_number, 
        request.future_months
    )
    
    if 'error' in predictions:
        raise HTTPException(status_code=404, detail=predictions['error'])
    
    # Format the response according to our schema
    response = {
        'account_number': request.account_number,
        'predictions': []
    }
    
    for pred in predictions['predictions']:
        # Create cost prediction
        cost_pred = pred.copy()
        for key in ['prediction_date', 'predicted_kwh', 'kwh_lower_bound', 'kwh_upper_bound']:
            if key in cost_pred:
                del cost_pred[key]
        
        # Create complete prediction
        complete_pred = {
            'prediction_date': pred['prediction_date'],
            'predicted_kwh': pred['predicted_kwh'],
            'lower_bound': pred.get('kwh_lower_bound', pred['predicted_kwh'] * 0.85),
            'upper_bound': pred.get('kwh_upper_bound', pred['predicted_kwh'] * 1.15),
            'avg_daily_temperature': pred.get('avg_daily_temperature'),
            'cost': cost_pred
        }
        
        response['predictions'].append(complete_pred)
    
    return response