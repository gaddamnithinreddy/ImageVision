"""
Streamlit app entry point for VisionAI.
This is the main file that Streamlit Cloud will run.
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Run the pure Streamlit implementation
import image_recognition_streamlit

if __name__ == "__main__":
    image_recognition_streamlit.main()

"""
Streamlit wrapper for the Flask application.
This file is required for Streamlit Cloud to recognize and deploy the application correctly.
"""

import streamlit as st
import subprocess
import os
import time
import requests
import threading
from streamlit.components.v1 import iframe

# Title for the Streamlit app
st.set_page_config(page_title="VisionAI - Image Recognition", layout="wide")
st.title("VisionAI - Image Recognition & Captioning")

# Add a note about the Flask backend
st.info("This application uses a Flask backend running on port 5000. Please wait for the server to start.")

# Function to start the Flask app
def start_flask_app():
    """Start the Flask application in the background."""
    try:
        # Change to the image_recognition_project directory
        os.chdir("image_recognition_project")
        
        # Start Flask app in background
        process = subprocess.Popen([
            "python", "app.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        os.chdir("..")  # Go back to the root directory
        return process
    except Exception as e:
        st.error(f"Failed to start Flask app: {str(e)}")
        return None

# Function to check if Flask app is running
def is_flask_running():
    """Check if the Flask application is running."""
    try:
        response = requests.get("http://localhost:5000", timeout=5)
        return response.status_code == 200
    except Exception as e:
        return False

# Start the Flask app
if 'flask_started' not in st.session_state:
    st.session_state.flask_started = True
    with st.spinner("Starting Flask backend server... This may take a few moments."):
        flask_process = start_flask_app()
        if flask_process:
            # Wait for the Flask app to start (up to 45 seconds)
            flask_ready = False
            for i in range(45):
                if is_flask_running():
                    st.success("Flask backend server started successfully!")
                    flask_ready = True
                    break
                time.sleep(1)
            
            if not flask_ready:
                st.warning("Flask server is still starting. You may need to refresh the page in a moment.")
        else:
            st.error("Failed to start Flask backend server.")

# Add a refresh button
if st.button("Refresh Application"):
    st.rerun()

# Check if Flask is running and display accordingly
if is_flask_running():
    st.success("Flask server is running! Loading application...")
    # Display the Flask app in an iframe
    iframe(
        "http://localhost:5000", 
        height=800, 
        scrolling=True
    )
else:
    st.warning("Flask server is still starting. Please wait a moment and click 'Refresh Application' above.")
    st.info("The Flask backend needs some time to initialize. This is normal during first startup.")

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