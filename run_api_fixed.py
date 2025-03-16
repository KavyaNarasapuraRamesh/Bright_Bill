# run_api_fixed.py
import sys
import os

# Add project root to Python path to solve import issues
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Make sure subdirectories are in the path too
for subdir in ['utils', 'services', 'scripts', 'ml_models']:
    path = os.path.join(current_dir, subdir)
    if path not in sys.path and os.path.exists(path):
        sys.path.insert(0, path)

# Create necessary directories
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/models", exist_ok=True)

# Initialize empty JSON file if it doesn't exist
combined_path = 'data/processed/combined_bills.json'
if not os.path.exists(combined_path):
    with open(combined_path, 'w') as f:
        f.write('[]')

# Now run the app
import uvicorn

if __name__ == "__main__":
    print("Starting Electricity Bill Analyzer API...")
    print("Python path:", sys.path)
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)