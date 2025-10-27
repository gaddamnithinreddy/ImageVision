"""
Streamlit wrapper for the Flask application.
This file is required for Streamlit Cloud to recognize and deploy the application correctly.
"""

import streamlit as st
import subprocess
import os
import time
import requests

# Import components
from streamlit.components.v1 import iframe

# Title for the Streamlit app
st.set_page_config(page_title="VisionAI - Image Recognition", layout="wide")
st.title("VisionAI - Image Recognition & Captioning")

# Add a note about the Flask backend
st.info("This application uses a Flask backend running on port 5000. Please wait for the server to start.")

# Function to start the Flask app
def start_flask_app():
    """Start the Flask application in the background."""
    # Change to the image_recognition_project directory
    os.chdir("image_recognition_project")
    
    # Start Flask app in background
    process = subprocess.Popen([
        "python", "app.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    os.chdir("..")  # Go back to the root directory
    return process

# Function to check if Flask app is running
def is_flask_running():
    """Check if the Flask application is running."""
    try:
        response = requests.get("http://localhost:5000", timeout=1)
        return response.status_code == 200
    except:
        return False

# Start the Flask app
if 'flask_started' not in st.session_state:
    st.session_state.flask_started = True
    with st.spinner("Starting Flask backend server..."):
        flask_process = start_flask_app()
        # Wait for the Flask app to start
        for i in range(30):  # Wait up to 30 seconds
            if is_flask_running():
                st.success("Flask backend server started successfully!")
                break
            time.sleep(1)
        else:
            st.error("Failed to start Flask backend server. Please check the logs.")

# Display the Flask app in an iframe
iframe("http://localhost:5000", height=800, scrolling=True)

# Add some information
st.markdown("""
---
### About this Application
This is a VisionAI application that uses Google's Gemini AI to recognize objects in images and generate social media captions.

### How to Use
1. Upload an image using the interface
2. Wait for the AI to analyze the image
3. View the recognized objects and confidence scores
4. Generate captions for social media platforms

### Technical Details
- Backend: Flask with Google Gemini AI
- Frontend: HTML/CSS/JavaScript
- Deployment: Streamlit Cloud wrapper
""")