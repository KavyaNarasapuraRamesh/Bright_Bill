from datetime import datetime

def parse_date(date_string):
    """Parse date string into ISO format YYYY-MM-DD"""
    try:
        # Try various formats
        formats = [
            "%B %d, %Y",  # January 19, 2024
            "%b %d, %Y",  # Jan 19, 2024
            "%b %d %Y",   # Jan 19 2024
            "%m/%d/%Y",   # 01/19/2024
            "%m-%d-%Y"    # 01-19-2024
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_string, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
                
        # If all formats fail, return None
        print(f"Could not parse date: {date_string}")
        return None
    except Exception as e:
        print(f"Error parsing date {date_string}: {e}")
        return None