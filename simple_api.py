# simple_api_stub.py - Modified with current bill data in both endpoints
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from datetime import datetime, timedelta

# Create FastAPI app
app = FastAPI(
    title="Electricity Bill Analyzer API",
    description="API for analyzing and predicting electricity bills",
    version="1.0.0"
)

# Configure CORS - allow your React app's origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create required directories
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Electricity Bill Analyzer API",
        "docs": "/docs",
        "endpoints": [
            "/api/upload",
            "/api/combined-prediction"
        ]
    }

# Upload endpoint that returns mock data
@app.post("/api/upload")
async def upload_bill(file: UploadFile = File(...), future_months: int = 3, print_enabled: bool = Form(False)):
    # Save the file to verify upload works
    file_path = f"data/raw/{file.filename}"
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    print(f"File successfully saved to {file_path}")
    print(f"Print enabled: {print_enabled}")
    
    # Generate future dates for predictions
    current_date = datetime.now()
    future_dates = [
        (current_date + timedelta(days=30 * (i + 1))).strftime("%Y-%m-%d")
        for i in range(future_months)
    ]
    
    # Return mock data with explicit current_bill field
    return {
        "user_info": {
            "account_number": "Mock-12345",
            "customer_name": "Test Customer"
        },
        "current_bill": {  # Explicit current_bill field for frontend
            "kwh_used": 925.5,
            "total_bill_amount": 138.83
        },
        "predictions": [
            {
                "prediction_date": future_dates[0],
                "predicted_kwh": 950.45,
                "total_bill_amount": 142.57,
                "utility_charges": 78.25,
                "supplier_charges": 64.32
            },
            {
                "prediction_date": future_dates[1],
                "predicted_kwh": 1050.20,
                "total_bill_amount": 157.53,
                "utility_charges": 86.54,
                "supplier_charges": 70.99
            },
            {
                "prediction_date": future_dates[2],
                "predicted_kwh": 980.75,
                "total_bill_amount": 147.11,
                "utility_charges": 80.91,
                "supplier_charges": 66.20
            }
        ],
        "ai_recommendations": [
            {
                "title": "Reduce Peak Hour Usage",
                "description": "Shift your major appliance usage to off-peak hours (typically after 8 PM) to benefit from lower rates."
            },
            {
                "title": "Check Refrigerator Efficiency",
                "description": "Your refrigerator usage pattern suggests it might be consuming more energy than normal. Consider checking the door seals and coil cleanliness."
            }
        ],
        "anomalies": [
            {
                "type": "usage_anomaly",
                "description": "Your electricity usage increased by 15% compared to the same period last year.",
                "severity": "medium"
            }
        ],
        "print_enabled": print_enabled  # Include print setting in response
    }

# Combined prediction endpoint - single month prediction with breakdown
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
    # If file is provided, save it
    if file:
        file_path = f"data/raw/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        print(f"File successfully saved to {file_path}")
    
    # Print appliance data
    print(f"Appliance data received:")
    print(f"- Air Conditioner: {air_conditioner} hours/day")
    print(f"- Refrigerator: {refrigerator} hours/day")
    print(f"- Water Heater: {water_heater} hours/day")
    print(f"- Clothes Dryer: {clothes_dryer} hours/day")
    print(f"- Washing Machine: {washing_machine} hours/day")
    print(f"- Household Size: {household_size}")
    print(f"- Home Size: {home_sqft} sqft")
    
    # Calculate some appliance breakdown values
    total_energy = 0
    breakdown = {}
    
    appliance_data = {
        'air_conditioner': {'avg_wattage': 1500, 'factor': 1.0},
        'refrigerator': {'avg_wattage': 150, 'factor': 24.0},
        'water_heater': {'avg_wattage': 4000, 'factor': 0.9},
        'clothes_dryer': {'avg_wattage': 3000, 'factor': 1.0},
        'washing_machine': {'avg_wattage': 500, 'factor': 1.0}
    }
    
    appliance_usage = {
        'air_conditioner': air_conditioner,
        'refrigerator': refrigerator,
        'water_heater': water_heater,
        'clothes_dryer': clothes_dryer,
        'washing_machine': washing_machine
    }
    
    for appliance, hours in appliance_usage.items():
        if appliance in appliance_data and hours > 0:
            app_info = appliance_data[appliance]
            energy = (app_info['avg_wattage'] * hours * app_info['factor'] * 30) / 1000
            total_energy += energy
    
    total_kwh = 980.25  # Example predicted usage
    estimated_cost = 147.04  # Example estimated cost
    
    # Calculate breakdown percentages
    for appliance, hours in appliance_usage.items():
        if appliance in appliance_data and hours > 0:
            app_info = appliance_data[appliance]
            energy = (app_info['avg_wattage'] * hours * app_info['factor'] * 30) / 1000
            percentage = (energy / total_energy * 100) if total_energy > 0 else 0
            
            breakdown[appliance] = {
                "hours_per_day": hours,
                "monthly_kwh": round(total_kwh * (percentage / 100), 2),
                "percentage": round(percentage, 1)
            }
    
    # Generate one month prediction
    next_month = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    
    # Return combined result optimized for pie chart visualization
    return {
        "user_info": {
            "account_number": "Mock-12345",
            "customer_name": "Test Customer"
        },
        "prediction": {
            "total_kwh": total_kwh,
            "estimated_cost": estimated_cost,
            "breakdown": breakdown,  # This is what will be used for the pie chart
            "prediction_date": next_month  # Adding date for reference
        },
        "current_bill": {
            "kwh_used": 925.5,
            "total_bill_amount": 138.83
        },
        # Include single prediction for consistency with frontend
        "predictions": [
            {
                "prediction_date": next_month,
                "predicted_kwh": total_kwh,
                "total_bill_amount": estimated_cost
            }
        ],
        "ai_recommendations": [
            {
                "title": "Optimize Air Conditioner Usage",
                "description": "Your air conditioner accounts for nearly 30% of your electricity usage. Consider setting it 2 degrees higher to save energy."
            },
            {
                "title": "Check For Phantom Power",
                "description": "Consider using smart power strips to eliminate standby power consumption from electronics."
            },
            {
                "title": "Refrigerator Efficiency",
                "description": "Your refrigerator runs 24/7. Consider cleaning the coils and checking the door seals to improve efficiency."
            }
        ]
    }

if __name__ == "__main__":
    print("Starting Electricity Bill Analyzer API on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)