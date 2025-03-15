from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import tempfile
import os
import sys
import json
import pandas as pd
from datetime import date, datetime

from utils.date_utils import standardize_date_format
from services.gemini_recommendation_service import GeminiRecommendationService


from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your services
recommendation_service = GeminiRecommendationService(api_key=api_key)
from scripts.direct_gemini_extraction import extract_bill_data
from services.prediction_service import PredictionService
from ml_models.anomaly_detector import AnomalyDetector

# Create FastAPI app
app = FastAPI(
    title="Electricity Bill Analyzer API",
    description="API for analyzing and predicting electricity bills",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, change to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define schemas
class BillBase(BaseModel):
    account_number: str
    bill_date: str
    kwh_used: float
    total_bill_amount: float

class PredictionRequest(BaseModel):
    account_number: str
    future_months: int = 3

class PredictionResponse(BaseModel):
    account_number: str
    predictions: List[Dict]

# Initialize services
# Note: extract_bill_data is a function, not a class, so we don't initialize it here
prediction_service = PredictionService()
anomaly_detector = AnomalyDetector()

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Electricity Bill Analyzer API",
        "docs": "/docs",
        "endpoints": [
            "/api/bills",
            "/api/predictions",
            "/api/anomalies",
            "/api/upload"
        ]
    }

# Get all bills
@app.get("/api/bills")
async def get_all_bills():
    try:
        with open('data/processed/combined_bills.json', 'r') as f:
            bills = json.load(f)
        return bills
    except Exception as e:
        return {"error": str(e)}

# Get specific bill
@app.get("/api/bills/{bill_id}")
async def get_bill(bill_id: int):
    try:
        with open('data/processed/combined_bills.json', 'r') as f:
            bills = json.load(f)
        
        if bill_id < 1 or bill_id > len(bills):
            raise HTTPException(status_code=404, detail="Bill not found")
        
        return bills[bill_id - 1]
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        return {"error": str(e)}

# Get predictions
@app.post("/api/predictions", response_model=PredictionResponse)
async def predict_bills(request: PredictionRequest):
    try:
        # Load historical data for this account
        with open('data/processed/combined_bills.json', 'r') as f:
            all_bills = json.load(f)
        
        # Filter for the requested account
        account_bills = [b for b in all_bills if b.get('account_number') == request.account_number]
        
        if not account_bills:
            raise HTTPException(status_code=404, detail=f"No data found for account {request.account_number}")
        
        # Convert to DataFrame
        df = pd.DataFrame(account_bills)
        
        # Convert date columns
        date_columns = ['bill_date', 'billing_start_date', 'billing_end_date', 'due_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Generate usage predictions
        usage_predictions = prediction_service.usage_predictor.predict(df, future_months=request.future_months)
        
        if usage_predictions is None:
            raise HTTPException(status_code=500, detail="Failed to generate usage predictions")
        
        # Generate cost predictions
        predictions = []
        for _, row in usage_predictions.iterrows():
            kwh_prediction = row['predicted_kwh']
            cost_prediction = prediction_service.cost_predictor.predict_cost(kwh_prediction)
            
            if cost_prediction:
                predictions.append({
                    "prediction_date": row['prediction_date'],
                    "predicted_kwh": kwh_prediction,
                    "lower_bound": row['lower_bound'],
                    "upper_bound": row['upper_bound'],
                    "avg_daily_temperature": row['avg_daily_temperature'],
                    "total_bill_amount": cost_prediction['total_bill_amount'],
                    "utility_charges": cost_prediction['utility_charges'],
                    "supplier_charges": cost_prediction['supplier_charges']
                })
        
        return {
            "account_number": request.account_number,
            "predictions": predictions
        }
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

# Get anomalies
@app.get("/api/anomalies/{bill_id}")
async def get_anomalies(bill_id: int):
    try:
        with open('data/processed/combined_bills.json', 'r') as f:
            bills = json.load(f)
        
        if bill_id < 1 or bill_id > len(bills):
            raise HTTPException(status_code=404, detail="Bill not found")
        
        bill = bills[bill_id - 1]
        
        # Detect anomalies
        anomalies = anomaly_detector.detect_anomalies(bill)
        
        return anomalies
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_bill(file: UploadFile = File(...), future_months: int = 3):
    try:
        # Save uploaded file
        file_path = f"data/raw/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Extract data from the bill using Gemini
        bill_data = extract_bill_data(file_path, api_key)
        
        if not bill_data:
            raise HTTPException(status_code=422, detail="Failed to extract data from bill")
        
        # Apply date preprocessing
        for col in ['bill_date', 'billing_start_date', 'billing_end_date', 'due_date']:
            if col in bill_data and bill_data[col]:
                bill_data[col] = standardize_date_format(bill_data[col])
        
        # Save to our database
        combined_path = 'data/processed/combined_bills.json'
        try:
            with open(combined_path, 'r') as f:
                existing_bills = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_bills = []
        
        existing_bills.append(bill_data)
        with open(combined_path, 'w') as f:
            json.dump(existing_bills, f, indent=2, default=str)
        
        # Get basic user info
        user_info = {
            "account_number": bill_data.get('account_number'),
            "customer_name": bill_data.get('customer_name')
        }
        
        # Detect anomalies
        anomalies = anomaly_detector.detect_anomalies(bill_data)
        
        # Generate predictions
        predictions = []
        try:
            # Create a dataframe from the bill data
            df = pd.DataFrame([bill_data])
            
            # Convert date columns to datetime
            date_columns = ['bill_date', 'billing_start_date', 'billing_end_date', 'due_date']
            for col in date_columns:
                if col in df.columns and df[col].iloc[0]:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Get usage predictions
            usage_predictions = prediction_service.usage_predictor.predict(df, future_months=future_months)
            
            if usage_predictions is not None:
                # For each predicted month
                for _, row in usage_predictions.iterrows():
                    # Get the predicted kWh
                    kwh_prediction = row['predicted_kwh']
                    
                    # Calculate the cost based on predicted kWh
                    cost_prediction = prediction_service.cost_predictor.predict_cost(kwh_prediction)
                    
                    if cost_prediction:
                        # Add prediction to results
                        predictions.append({
                            "prediction_date": row['prediction_date'],
                            "predicted_kwh": kwh_prediction,
                            "total_bill_amount": cost_prediction['total_bill_amount'],
                            "utility_charges": cost_prediction['utility_charges'],
                            "supplier_charges": cost_prediction['supplier_charges']
                        })
        except Exception as e:
            print(f"Error generating predictions: {str(e)}")
        
        # Generate AI recommendations using Gemini
        ai_recommendations = recommendation_service.generate_insights(bill_data, predictions, anomalies)
        
        # Return response with all components
        response = {
            "user_info": user_info,
            "predictions": predictions,
            "ai_recommendations": ai_recommendations
        }
        
        # Add anomalies if any were found
        if anomalies:
            response["anomalies"] = anomalies
        else:
            response["anomalies"] = [{"type": "info", "description": "No anomalies detected in your bill.", "severity": "low"}]
        
        return response
        
    except Exception as e:
        print(f"Error in upload endpoint: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
# Helper function for AI recommendations
def generate_recommendations(bill_data, anomalies):
    """Generate recommendations based on bill data and anomalies"""
    recommendations = []
    
    # Basic recommendations based on usage
    kwh_used = bill_data.get('kwh_used', 0)
    avg_daily_usage = bill_data.get('avg_daily_usage', 0)
    
    if kwh_used > 1000:
        recommendations.append({
            "category": "high_usage",
            "description": "Your electricity usage is high. Consider energy-efficient appliances and turning off lights when not in use."
        })
    
    if avg_daily_usage > 30:
        recommendations.append({
            "category": "daily_usage",
            "description": "Your daily electricity usage is above average. Check for devices that may be running unnecessarily."
        })
    
    # Recommendations based on anomalies
    for anomaly in anomalies:
        if anomaly.get('type') == 'usage_anomaly':
            recommendations.append({
                "category": "anomaly_related",
                "description": f"We detected unusual usage: {anomaly.get('description')}. Consider checking for malfunctioning appliances or changes in usage patterns."
            })
        elif anomaly.get('type') == 'rate_anomaly':
            recommendations.append({
                "category": "pricing",
                "description": f"We noticed an issue with your rates: {anomaly.get('description')}. Consider reviewing your supplier contract or exploring other providers."
            })
    
    # Seasonal recommendations
    bill_date = bill_data.get('bill_date')
    if bill_date:
        if isinstance(bill_date, str):
            try:
                bill_date = datetime.strptime(bill_date, '%Y-%m-%d')
            except:
                try:
                    bill_date = datetime.strptime(bill_date, '%B %d, %Y')
                except:
                    bill_date = None
        
        if bill_date:
            month = bill_date.month
            if month in [12, 1, 2]:  # Winter
                recommendations.append({
                    "category": "seasonal",
                    "description": "Winter months typically have higher electricity usage. Consider lowering your thermostat by a few degrees and using energy-efficient space heaters where needed."
                })
            elif month in [6, 7, 8]:  # Summer
                recommendations.append({
                    "category": "seasonal",
                    "description": "During summer months, air conditioning can increase electricity bills. Consider using fans and keeping blinds closed during the day to reduce cooling needs."
                })
    
    return recommendations