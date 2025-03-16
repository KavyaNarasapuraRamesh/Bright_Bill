# run_api.py
import sys
import os

# Add the current directory to the Python path
# This is crucial for proper module resolution
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# If your main.py is inside an 'api' folder, make sure that folder
# is in the Python path as well
# Don't include the FastAPI and CORS imports here since they're in main.py

if __name__ == "__main__":
    # Import uvicorn only after setting the path
    import uvicorn
    
    # Use the correct module path to your app
    # If main.py is in a folder called 'api' and app is defined in main.py
    uvicorn.run("api.main:app", host="127.0.0.1", port=8080, reload=True)