"""
Pure Streamlit implementation of the VisionAI Image Recognition application.
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Run the pure Streamlit implementation
import image_recognition_streamlit

if __name__ == "__main__":
    image_recognition_streamlit.main()