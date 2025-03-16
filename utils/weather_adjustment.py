"""
Weather adjustment utilities for electricity bill analysis.
This module provides functions for calculating weather-adjusted
electricity usage to provide fair comparisons across different periods.
"""

import numpy as np
from datetime import datetime

def calculate_hdd_cdd(temperature, base_temp=65):
    """
    Calculate Heating Degree Days (HDD) and Cooling Degree Days (CDD)
    
    Args:
        temperature: Average daily temperature (°F)
        base_temp: Base temperature for HDD/CDD calculation (default: 65°F)
        
    Returns:
        Tuple of (HDD, CDD) values
    """
    if temperature < base_temp:
        return (base_temp - temperature, 0)
    else:
        return (0, temperature - base_temp)

def weather_adjust_usage(current_usage, current_temp, reference_temp, 
                         heating_factor=0.1, cooling_factor=0.15):
    """
    Adjust electricity usage based on temperature differences
    
    Args:
        current_usage: The actual usage (kWh) to adjust
        current_temp: The average temperature during current usage period
        reference_temp: The temperature to normalize to
        heating_factor: kWh per degree day for heating (default: 0.1)
        cooling_factor: kWh per degree day for cooling (default: 0.15)
        
    Returns:
        Weather-adjusted usage amount
    """
    # Calculate degree days for both periods
    current_hdd, current_cdd = calculate_hdd_cdd(current_temp)
    reference_hdd, reference_cdd = calculate_hdd_cdd(reference_temp)
    
    # Calculate adjustment based on difference in degree days
    heating_adjustment = (reference_hdd - current_hdd) * heating_factor
    cooling_adjustment = (reference_cdd - current_cdd) * cooling_factor
    
    # Apply adjustments to current usage
    adjusted_usage = current_usage + heating_adjustment + cooling_adjustment
    
    # Ensure adjusted usage is positive
    return max(adjusted_usage, 0)

def compare_bills_weather_adjusted(current_bill, previous_bill):
    """
    Compare two bills with weather adjustment to show true usage changes
    
    Args:
        current_bill: Dictionary with current bill data
        previous_bill: Dictionary with previous bill data
        
    Returns:
        Dictionary with comparison metrics
    """
    # Extract key values
    current_usage = current_bill.get('kwh_used', 0)
    previous_usage = previous_bill.get('kwh_used', 0)
    
    current_temp = current_bill.get('avg_daily_temperature', 65)
    previous_temp = previous_bill.get('avg_daily_temperature', 65)
    
    # Calculate weather-adjusted previous usage
    weather_adjusted_previous = weather_adjust_usage(
        previous_usage, previous_temp, current_temp
    )
    
    # Calculate raw and weather-adjusted differences
    raw_diff = current_usage - previous_usage
    raw_diff_percent = (raw_diff / previous_usage * 100) if previous_usage > 0 else 0
    
    adjusted_diff = current_usage - weather_adjusted_previous
    adjusted_diff_percent = (adjusted_diff / weather_adjusted_previous * 100) if weather_adjusted_previous > 0 else 0
    
    # Calculate how much of the change was weather-related
    weather_impact = raw_diff - adjusted_diff
    weather_impact_percent = (weather_impact / previous_usage * 100) if previous_usage > 0 else 0
    
    # Return comparison metrics
    return {
        'current_usage': current_usage,
        'previous_usage': previous_usage,
        'weather_adjusted_previous': round(weather_adjusted_previous, 2),
        'raw_diff': round(raw_diff, 2),
        'raw_diff_percent': round(raw_diff_percent, 1),
        'adjusted_diff': round(adjusted_diff, 2),
        'adjusted_diff_percent': round(adjusted_diff_percent, 1),
        'weather_impact': round(weather_impact, 2),
        'weather_impact_percent': round(weather_impact_percent, 1),
        'current_temperature': current_temp,
        'previous_temperature': previous_temp
    }

def get_weather_impact_text(comparison):
    """
    Generate user-friendly text explaining weather impact
    
    Args:
        comparison: The output from compare_bills_weather_adjusted
        
    Returns:
        String with user-friendly explanation
    """
    weather_impact = comparison['weather_impact']
    weather_impact_percent = comparison['weather_impact_percent']
    
    if abs(weather_impact) < 10 or abs(weather_impact_percent) < 5:
        return "Weather had minimal impact on your usage this month."
    
    if weather_impact > 0:
        return f"Weather conditions increased your usage by {abs(weather_impact):.1f} kWh ({abs(weather_impact_percent):.1f}%) compared to last month. Without this weather effect, your usage would have been lower."
    else:
        return f"Weather conditions decreased your usage by {abs(weather_impact):.1f} kWh ({abs(weather_impact_percent):.1f}%) compared to last month. Without this weather effect, your usage would have been higher."