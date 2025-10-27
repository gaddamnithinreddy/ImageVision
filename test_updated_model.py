import os
from dotenv import load_dotenv
from image_recognition_project.models import GeminiModel

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv('GEMINI_API_KEY')

if not api_key:
    print("ERROR: GEMINI_API_KEY not found in environment variables")
else:
    print("API Key found, initializing model...")
    try:
        model = GeminiModel(api_key=api_key, model_name="gemini-2.0-flash")
        print("SUCCESS: Gemini model initialized successfully")
        
        # Test the model with a simple prompt
        print("Testing model with a simple prompt...")
        try:
            response = model.model.generate_content("Say hello world")
            print(f"Model response: {response.text}")
        except Exception as e:
            print(f"Error testing model: {e}")
    except Exception as e:
        print(f"ERROR: Failed to initialize Gemini model: {e}")