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
import pickle
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form

from utils.date_utils import standardize_date_format
from services.gemini_recommendation_service import GeminiRecommendationService
from utils.data_manager import save_bill_data_to_history, retrain_models_with_history

from dotenv import load_dotenv
import os

load_dotenv()

#api_key = os.getenv("GEMINI_API_KEY")
api_key = "AIzaSyD7sb6xFqv4lWvVW9KhtHVSbJ9tG4L6E2E"
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your services
recommendation_service = GeminiRecommendationService(api_key=api_key)
from scripts.direct_gemini_extraction import extract_bill_data
from services.prediction_service import PredictionService
from ml_models.anomaly_detector import AnomalyDetector
from utils.weather_adjustment import compare_bills_weather_adjusted, get_weather_impact_text
from utils.gamification import (
    load_user_gamification_data, 
    check_achievements, 
    generate_gamification_summary
)


# Load pre-trained ML models
def load_ml_models():
    models = {}
    try:
        models_dir = 'models'
        
        # Bill prediction model
        if os.path.exists(f'{models_dir}/bill_predictor_model.pkl'):
            with open(f'{models_dir}/bill_predictor_model.pkl', 'rb') as f:
                models['bill_model'] = pickle.load(f)
            with open(f'{models_dir}/bill_predictor_scaler.pkl', 'rb') as f:
                models['bill_scaler'] = pickle.load(f)
                
        # Appliance prediction model
        if os.path.exists(f'{models_dir}/appliance_predictor_model.pkl'):
            with open(f'{models_dir}/appliance_predictor_model.pkl', 'rb') as f:
                models['appliance_model'] = pickle.load(f)
            with open(f'{models_dir}/appliance_predictor_scaler.pkl', 'rb') as f:
                models['appliance_scaler'] = pickle.load(f)
                
        # Combined prediction model
        if os.path.exists(f'{models_dir}/combined_predictor_model.pkl'):
            with open(f'{models_dir}/combined_predictor_model.pkl', 'rb') as f:
                models['combined_model'] = pickle.load(f)
            with open(f'{models_dir}/combined_predictor_scaler.pkl', 'rb') as f:
                models['combined_scaler'] = pickle.load(f)
        
        print(f"Loaded {len(models)//2} ML models successfully")
        return models
    except Exception as e:
        print(f"Error loading ML models: {str(e)}")
        return {}

# Load models at startup
ml_models = load_ml_models()

# Create FastAPI app
app = FastAPI(
    title="Electricity Bill Analyzer API",
    description="API for analyzing and predicting electricity bills",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # In production, change to your frontend domain
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

class ApplianceUsageRequest(BaseModel):
    air_conditioner: float = 0
    refrigerator: float = 24
    water_heater: float = 0
    clothes_dryer: float = 0
    washing_machine: float = 0
    household_size: int = 3
    home_sqft: int = 1800

class CombinedPredictionRequest(BaseModel):
    bill_id: int
    air_conditioner: float = 0
    refrigerator: float = 24
    water_heater: float = 0
    clothes_dryer: float = 0
    washing_machine: float = 0
    household_size: int = 3
    home_sqft: int = 1800

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
            "/api/upload",
            "/api/appliances",
            "/api/combined-prediction"
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
async def upload_bill(file: UploadFile = File(...), future_months: int = 3, print_enabled: bool = Form(False)):
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
        
        # Save only the extracted bill data to the historical dataset
        save_bill_data_to_history(bill_data)
        
        # Save the bill data to our combined_bills.json file
        combined_path = 'data/processed/combined_bills.json'
        try:
            with open(combined_path, 'r') as f:
                existing_bills = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_bills = []
        
        existing_bills.append(bill_data)
        with open(combined_path, 'w') as f:
            json.dump(existing_bills, f, indent=2, default=str)
        
        # Check if retraining is needed - do this after saving the data
        if os.path.exists('data/models/retrain_needed.txt'):
            # For a hackathon, this can be done synchronously
            # In production, this should be a background task
            print("Retraining models with latest historical data")
            retrain_models_with_history()
        
        # Get basic user info
        user_info = {
            "account_number": bill_data.get('account_number'),
            "customer_name": bill_data.get('customer_name')
        }
        
        # Extract current usage and bill amount
        current_bill = {
            "kwh_used": bill_data.get('kwh_used'),
            "total_bill_amount": bill_data.get('total_bill_amount')
        }
        
        # Detect anomalies
        anomalies = anomaly_detector.detect_anomalies(bill_data)
        
        # Generate predictions directly from this bill only
        predictions = prediction_service.predict_from_single_bill(bill_data, future_months=future_months)
        
        # Generate AI recommendations using Gemini
        ai_recommendations = recommendation_service.generate_insights(bill_data, predictions, anomalies)
        
        # Return response with all components
        response = {
            "user_info": user_info,
            "current_bill": current_bill,
            "predictions": predictions,  # Include predictions in response but don't save them
            "ai_recommendations": ai_recommendations,
            "print_enabled": print_enabled
        }
        
        # Add anomalies if any were found
        if anomalies:
            response["anomalies"] = anomalies
        else:
            response["anomalies"] = [{"type": "info", "description": "No anomalies detected in your bill.", "severity": "low"}]
        # Add weather comparison if previous bill exists
        weather_comparison = None
        account_number = bill_data.get('account_number')

        try:
            # Find previous bill for this account
            with open('data/processed/combined_bills.json', 'r') as f:
                all_bills = json.load(f)
            
            previous_bills = [b for b in all_bills 
                            if b.get('account_number') == account_number
                            and b.get('bill_date') < bill_data.get('bill_date')]
            
            if previous_bills:
                # Get most recent previous bill
                previous_bill = sorted(
                    previous_bills, 
                    key=lambda b: b.get('bill_date', '2000-01-01'),
                    reverse=True
                )[0]
                
                # Calculate weather-adjusted comparison
                weather_comparison = compare_bills_weather_adjusted(bill_data, previous_bill)
                weather_comparison['explanation'] = get_weather_impact_text(weather_comparison)
                
                # Add gamification
                user_data = load_user_gamification_data(account_number)
                updated_user_data, new_achievements = check_achievements(bill_data, previous_bill, user_data)
                gamification_summary = generate_gamification_summary(updated_user_data, new_achievements)
                
                # Add to response
                response["weather_comparison"] = weather_comparison
                response["gamification"] = gamification_summary
            else:
                # First bill - just add gamification
                user_data = load_user_gamification_data(account_number)
                updated_user_data, new_achievements = check_achievements(bill_data, None, user_data)
                gamification_summary = generate_gamification_summary(updated_user_data, new_achievements)
                
                # Add to response
                response["gamification"] = gamification_summary
                
        except Exception as comparison_error:
            print(f"Error generating comparison data: {str(comparison_error)}")
            # Non-fatal error, continue without comparison
        
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

    # Add new endpoint for manual retraining
@app.post("/api/retrain")
async def retrain_models():
    """Manually trigger model retraining"""
    try:
        success = retrain_models_with_history()
        if success:
            return {"message": "Models retrained successfully with historical data"}
        else:
            return {"message": "Retraining skipped - not enough historical data"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retraining models: {str(e)}")

@app.post("/api/appliances")
async def predict_from_appliances(request: ApplianceUsageRequest):
    """Predict electricity usage based on appliance usage"""
    try:
        if 'appliance_model' not in ml_models or 'appliance_scaler' not in ml_models:
            raise HTTPException(status_code=500, detail="Appliance prediction model not loaded")
        
        # Create feature array with correct column names
        features = pd.DataFrame([{
            'air_conditioner_hours': request.air_conditioner,
            'refrigerator_hours': request.refrigerator,
            'electric_water_heater_hours': request.water_heater,
            'clothes_dryer_hours': request.clothes_dryer,
            'washing_machine_hours': request.washing_machine,
            'household_size': request.household_size,
            'home_sqft': request.home_sqft
        }])
        
        # Scale features
        features_scaled = ml_models['appliance_scaler'].transform(features)
        
        # Predict
        kwh_prediction = ml_models['appliance_model'].predict(features_scaled)[0]
        estimated_cost = kwh_prediction * 0.15  # $0.15 per kWh
        
        # Calculate breakdown
        appliance_data = {
            'air_conditioner': {'avg_wattage': 1500, 'factor': 1.0},
            'refrigerator': {'avg_wattage': 150, 'factor': 24.0},
            'water_heater': {'avg_wattage': 4000, 'factor': 0.9},
            'clothes_dryer': {'avg_wattage': 3000, 'factor': 1.0},
            'washing_machine': {'avg_wattage': 500, 'factor': 1.0}
        }
        
        total_energy = 0
        breakdown = {}
        
        appliance_usage = {
            'air_conditioner': request.air_conditioner,
            'refrigerator': request.refrigerator,
            'water_heater': request.water_heater,
            'clothes_dryer': request.clothes_dryer,
            'washing_machine': request.washing_machine
        }
        
        for appliance, hours in appliance_usage.items():
            if appliance in appliance_data and hours > 0:
                app_info = appliance_data[appliance]
                energy = (app_info['avg_wattage'] * hours * app_info['factor'] * 30) / 1000
                total_energy += energy
        
        # Distribute predicted energy proportionally
        if total_energy > 0:
            for appliance, hours in appliance_usage.items():
                if appliance in appliance_data and hours > 0:
                    app_info = appliance_data[appliance]
                    energy = (app_info['avg_wattage'] * hours * app_info['factor'] * 30) / 1000
                    proportion = energy / total_energy
                    
                    breakdown[appliance] = {
                        "hours_per_day": hours,
                        "monthly_kwh": round(kwh_prediction * proportion, 2),
                        "percentage": round(proportion * 100, 1)
                    }
        
        # Generate AI recommendations
        ai_recommendations = generate_appliance_recommendations(appliance_usage, kwh_prediction, breakdown)
        
        return {
            "prediction": {
                "total_kwh": round(kwh_prediction, 2),
                "estimated_cost": round(estimated_cost, 2),
                "breakdown": breakdown
            },
            "ai_recommendations": ai_recommendations
        }
    except Exception as e:
        print(f"Error in appliance prediction: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form

# Add this endpoint to your FastAPI app:
@app.post("/api/combined-prediction")
async def predict_combined(
    file: UploadFile = File(None),
    bill_id: int = Form(None),
    air_conditioner: float = Form(0),
    refrigerator: float = Form(24),
    water_heater: float = Form(0),
    clothes_dryer: float = Form(0),
    washing_machine: float = Form(0),
    household_size: int = Form(3),
    home_sqft: int = Form(1800)
):
    """Predict electricity usage using combined bill and appliance data"""
    try:
        if 'combined_model' not in ml_models or 'combined_scaler' not in ml_models:
            raise HTTPException(status_code=500, detail="Combined prediction model not loaded")
        
        bill_data = None
        
        # Handle bill data from either file upload or bill_id
        if file:
            # Save uploaded file
            file_path = f"data/raw/{file.filename}"
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Extract data from the bill using Gemini
            bill_data = extract_bill_data(file_path, api_key)
            
            if not bill_data:
                raise HTTPException(status_code=422, detail="Failed to extract data from bill")
                
            # Save extracted bill data to history
            save_bill_data_to_history(bill_data)
            
            # Apply date preprocessing
            for col in ['bill_date', 'billing_start_date', 'billing_end_date', 'due_date']:
                if col in bill_data and bill_data[col]:
                    bill_data[col] = standardize_date_format(bill_data[col])
            
            # Save the raw bill data to our database
            combined_path = 'data/processed/combined_bills.json'
            try:
                with open(combined_path, 'r') as f:
                    existing_bills = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing_bills = []
            
            existing_bills.append(bill_data)
            with open(combined_path, 'w') as f:
                json.dump(existing_bills, f, indent=2, default=str)
                
        elif bill_id:
            # Get bill data from database using bill_id
            try:
                with open('data/processed/combined_bills.json', 'r') as f:
                    bills = json.load(f)
                
                if bill_id < 1 or bill_id > len(bills):
                    raise HTTPException(status_code=404, detail="Bill not found")
                
                bill_data = bills[bill_id - 1]
            except Exception as e:
                raise HTTPException(status_code=404, detail=f"Error retrieving bill: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Either file or bill_id must be provided")
        
        # Extract month from bill date
        bill_date = None
        month = datetime.now().month  # Default to current month
        
        if 'bill_date' in bill_data:
            try:
                if isinstance(bill_data['bill_date'], str):
                    bill_date = pd.to_datetime(bill_data['bill_date'], errors='coerce')
                    if bill_date is not None:
                        month = bill_date.month
            except:
                pass
        
        # Create feature array - use data from bill if appliance data not provided
        appliance_usage = {
            'air_conditioner': air_conditioner,
            'refrigerator': refrigerator,
            'water_heater': water_heater,
            'clothes_dryer': clothes_dryer,
            'washing_machine': washing_machine
        }
        
        # Use bill data for household info if not provided
        household_size_value = household_size if household_size is not None else bill_data.get('household_size', 3)
        home_sqft_value = home_sqft if home_sqft is not None else bill_data.get('home_sqft', 1800)
        
        features = pd.DataFrame([{
            'household_size': household_size_value,
            'home_sqft': home_sqft_value,
            'avg_daily_temperature': bill_data.get('avg_daily_temperature', 70),
            'month': month,
            'air_conditioner_hours': appliance_usage['air_conditioner'],
            'refrigerator_hours': appliance_usage['refrigerator'],
            'electric_water_heater_hours': appliance_usage['water_heater'],
            'clothes_dryer_hours': appliance_usage['clothes_dryer'],
            'washing_machine_hours': appliance_usage['washing_machine']
        }])
        
        # Scale features
        features_scaled = ml_models['combined_scaler'].transform(features)
        
        # Predict
        kwh_prediction = ml_models['combined_model'].predict(features_scaled)[0]
        estimated_cost = kwh_prediction * 0.15  # $0.15 per kWh
        
        # Calculate breakdown
        appliance_data = {
            'air_conditioner': {'avg_wattage': 1500, 'factor': 1.0},
            'refrigerator': {'avg_wattage': 150, 'factor': 24.0},
            'water_heater': {'avg_wattage': 4000, 'factor': 0.9},
            'clothes_dryer': {'avg_wattage': 3000, 'factor': 1.0},
            'washing_machine': {'avg_wattage': 500, 'factor': 1.0}
        }
        
        total_energy = 0
        breakdown = {}
        
        for appliance, hours in appliance_usage.items():
            if appliance in appliance_data and hours > 0:
                app_info = appliance_data[appliance]
                energy = (app_info['avg_wattage'] * hours * app_info['factor'] * 30) / 1000
                total_energy += energy
        
        # Distribute predicted energy proportionally
        if total_energy > 0:
            for appliance, hours in appliance_usage.items():
                if appliance in appliance_data and hours > 0:
                    app_info = appliance_data[appliance]
                    energy = (app_info['avg_wattage'] * hours * app_info['factor'] * 30) / 1000
                    proportion = energy / total_energy
                    
                    breakdown[appliance] = {
                        "hours_per_day": hours,
                        "monthly_kwh": round(kwh_prediction * proportion, 2),
                        "percentage": round(proportion * 100, 1)
                    }
        
        # Generate future predictions (3 months) similar to bill upload - but don't save them
        predictions = []
        future_months = 3
        
        try:
            # Create base prediction
            base_prediction = {
                "prediction_date": datetime.now().strftime("%Y-%m-%d"),
                "predicted_kwh": round(kwh_prediction, 2),
                "total_bill_amount": round(estimated_cost, 2),
                "utility_charges": round(estimated_cost * 0.55, 2),  # Example split
                "supplier_charges": round(estimated_cost * 0.45, 2)  # Example split
            }
            
            # Add first month prediction
            predictions.append(base_prediction)
            
            # Add additional months with slight variations
            for i in range(1, future_months):
                # Calculate future date
                future_date = (datetime.now() + timedelta(days=30 * i)).strftime("%Y-%m-%d")
                
                # Add some variation to predictions (5% increase per month as an example)
                variation_factor = 1 + (0.05 * i)
                
                future_kwh = round(kwh_prediction * variation_factor, 2)
                future_cost = round(future_kwh * 0.15, 2)
                
                predictions.append({
                    "prediction_date": future_date,
                    "predicted_kwh": future_kwh,
                    "total_bill_amount": future_cost,
                    "utility_charges": round(future_cost * 0.55, 2),
                    "supplier_charges": round(future_cost * 0.45, 2)
                })
        except Exception as e:
            print(f"Error generating multiple predictions: {str(e)}")
            # Still continue even if this fails
        
        # Generate AI recommendations
        ai_recommendations = generate_combined_recommendations(bill_data, appliance_usage, kwh_prediction, breakdown)
        
        # Return combined result with current usage/bill info and 3-month predictions
        return {
            "user_info": {
                "account_number": bill_data.get('account_number'),
                "customer_name": bill_data.get('customer_name')
            },
            "prediction": {
                "total_kwh": round(kwh_prediction, 2),
                "estimated_cost": round(estimated_cost, 2),
                "breakdown": breakdown
            },
            "current_bill": {
                "kwh_used": bill_data.get('kwh_used'),
                "total_bill_amount": bill_data.get('total_bill_amount')
            },
            "predictions": predictions,  # Include in response but don't save to database
            "ai_recommendations": ai_recommendations
        }
        
    except Exception as e:
        print(f"Error in combined prediction: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def generate_combined_recommendations(bill_data, appliance_usage, kwh_prediction, breakdown):
    """Generate AI recommendations based on both bill and appliance data"""
    try:
        if not api_key:
            return []
            
        # Format bill data
        bill_info = f"""
        Current Bill:
        - Account: {bill_data.get('account_number')}
        - Date: {bill_data.get('bill_date')}
        - Usage: {bill_data.get('kwh_used')} kWh
        - Amount: ${bill_data.get('total_bill_amount')}
        """
        
        # Format appliance data
        appliance_data = "\n".join([
            f"- {appliance.replace('_', ' ').title()}: {hours} hours/day ({breakdown.get(appliance, {}).get('percentage', 0)}% of usage)" 
            for appliance, hours in appliance_usage.items() if hours > 0
        ])
        
        # Format prediction data
        prediction_data = f"""
        Predicted Usage: {kwh_prediction} kWh
        Estimated Cost: ${kwh_prediction * 0.15:.2f}
        Difference from current: {kwh_prediction - bill_data.get('kwh_used', 0):.2f} kWh
        """
        
        prompt = f"""
        As an energy efficiency expert, analyze this electricity usage data and provide personalized recommendations:
        
        {bill_info}
        
        Appliance Usage:
        {appliance_data}
        
        {prediction_data}
        
        Based on both the bill history and appliance usage patterns, provide 3 specific, actionable recommendations
        to reduce electricity consumption and save money. Consider seasonal factors and usage patterns.
        
        Format your response as a JSON array of objects with 'title' and 'description' fields.
        """
        
        # Call Gemini for personalized recommendations
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        
        # Extract JSON from response
        import re
        import json
        json_pattern = r'(\[[\s\S]*\])'
        match = re.search(json_pattern, response.text)
        
        if match:
            return json.loads(match.group(1))
        else:
            # Fallback recommendations
            return [
                {
                    "title": "Optimize Your Highest Energy Consumer",
                    "description": "Based on your appliance usage, reducing your highest energy consuming appliance by just 10% could save you significant money on your bill."
                },
                {
                    "title": "Consider Programmable Thermostats",
                    "description": "Installing a programmable thermostat can save up to 10% annually on heating and cooling by automatically adjusting temperatures when you're away."
                }
            ]
    except Exception as e:
        print(f"Error generating combined recommendations: {str(e)}")
        return []
    

@app.get("/api/weather-comparison/{bill_id}")
async def get_weather_comparison(bill_id: int):
    """Get weather-adjusted comparison between this bill and the previous one"""
    try:
        # Load bills
        with open('data/processed/combined_bills.json', 'r') as f:
            bills = json.load(f)
        
        if bill_id < 1 or bill_id > len(bills):
            raise HTTPException(status_code=404, detail="Bill not found")
        
        current_bill = bills[bill_id - 1]
        
        # Find previous bill for this account
        account_number = current_bill.get('account_number')
        previous_bills = [b for b in bills if b.get('account_number') == account_number 
                           and b.get('bill_date') < current_bill.get('bill_date')]
        
        if not previous_bills:
            return {
                "status": "no_previous_bill",
                "message": "No previous bill found for comparison"
            }
        
        # Get most recent previous bill
        previous_bill = sorted(
            previous_bills, 
            key=lambda b: b.get('bill_date', '2000-01-01'),
            reverse=True
        )[0]
        
        # Calculate weather-adjusted comparison
        comparison = compare_bills_weather_adjusted(current_bill, previous_bill)
        
        # Add explanation text
        comparison['explanation'] = get_weather_impact_text(comparison)
        comparison['status'] = "success"
        
        return comparison
        
    except Exception as e:
        print(f"Error in weather comparison: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/gamification/{account_number}")
async def get_gamification_data(account_number: str):
    """Get gamification data for a user"""
    try:
        user_data = load_user_gamification_data(account_number)
        summary = generate_gamification_summary(user_data)
        return summary
        
    except Exception as e:
        print(f"Error getting gamification data: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))



