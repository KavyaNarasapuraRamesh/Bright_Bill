# utils/date_utils.py
from datetime import datetime
from dateutil import parser

def standardize_date_format(date_str):
    """Convert various date formats to YYYY-MM-DD"""
    if not date_str:
        return None
        
    try:
        # Try different common formats
        for fmt in ['%B %d, %Y', '%b %d, %Y', '%m/%d/%Y', '%Y-%m-%d', '%d-%m-%Y']:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
                
        # If still here, try with dateutil parser
        dt = parser.parse(date_str)
        return dt.strftime('%Y-%m-%d')
    except:
        return None