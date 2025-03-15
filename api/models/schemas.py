from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date, datetime

class BillBase(BaseModel):
    account_number: str
    bill_date: date
    billing_start_date: date
    billing_end_date: date
    days_in_billing_period: int
    kwh_used: float
    avg_daily_temperature: Optional[float] = None

class BillCreate(BillBase):
    meter_start_value: float
    meter_end_value: float
    avg_daily_usage: float
    total_bill_amount: float
    utility_price_to_compare: float
    supplier_rate: float
    customer_charge: float
    distribution_related_component: float
    cost_recovery_charges: float
    consumer_rate_credit: float
    distribution_credit: Optional[float] = 0.0
    non_standard_credit: Optional[float] = 0.0
    utility_charges: float
    supplier_charges: float

class BillResponse(BillCreate):
    id: int
    
    class Config:
        orm_mode = True

class PredictionRequest(BaseModel):
    account_number: str
    future_months: int = Field(3, description="Number of months to predict")
    
class UsagePrediction(BaseModel):
    prediction_date: date
    predicted_kwh: float
    lower_bound: float
    upper_bound: float
    avg_daily_temperature: Optional[float] = None

class CostPrediction(BaseModel):
    customer_charge: float
    distribution_related_component: float
    cost_recovery_charges: float
    consumer_rate_credit: float
    distribution_credit: float
    non_standard_credit: float
    utility_charges: float
    supplier_charges: float
    total_bill_amount: float

class CompletePrediction(UsagePrediction):
    cost: CostPrediction

class PredictionResponse(BaseModel):
    account_number: str
    predictions: List[CompletePrediction]

class AnomalyResponse(BaseModel):
    bill_id: int
    type: str
    description: str
    severity: str