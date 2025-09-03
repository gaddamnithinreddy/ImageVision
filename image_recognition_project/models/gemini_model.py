import os
import google.generativeai as genai
from PIL import Image
import io
import base64

class GeminiModel:
    def __init__(self, api_key=None, model_name="gemini-1.5-flash-latest"):
        """
        Initialize the Gemini model with API key.
        
        Args:
            api_key (str): Your Google AI API key. If None, will look for GOOGLE_API_KEY environment variable.
            model_name (str): Name of the Gemini model to use.
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("API key is required. Please set GOOGLE_API_KEY environment variable or pass it as an argument.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        
    def preprocess_image(self, pil_image):
        """
        Convert PIL Image to a format suitable for Gemini API.
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            dict: Processed image data
        """
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        return {
            'mime_type': 'image/jpeg',
            'data': base64.b64encode(img_byte_arr).decode('utf-8')
        }
    
    def predict(self, image_data, prompt=None, top_k=5):
        """
        Make a prediction using the Gemini model.
        
        Args:
            image_data: Image data from preprocess_image
            prompt (str): Optional custom prompt
            top_k (int): Number of predictions to return
            
        Returns:
            list: List of predictions with confidence scores
        """
        if prompt is None:
            prompt = f"""
            Analyze this image and provide the top {top_k} most likely objects or scenes in the image.
            For each prediction, provide:
            1. The name of the object/scene
            2. A confidence score between 0 and 1
            
            Format the response as a JSON array of objects with 'class' and 'confidence' keys.
            Example response:
            [
                {{"class": "dog", "confidence": 0.95}},
                {{"class": "cat", "confidence": 0.85}}
            ]
            """
        
        try:
            # Convert image data to PIL Image
            img_bytes = base64.b64decode(image_data['data'])
            
            # Generate content with the image and prompt
            response = self.model.generate_content([
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": image_data['data']}}
            ])
            
            # Parse the response text as JSON
            try:
                import json
                # Extract the text from the response
                response_text = response.text
                
                # Clean up the response text to make it valid JSON
                response_text = response_text.strip()
                if '```json' in response_text:
                    response_text = response_text.split('```json')[1].split('```')[0].strip()
                elif '```' in response_text:
                    response_text = response_text.split('```')[1].strip()
                
                # Parse the JSON response
                predictions = json.loads(response_text)
                
                # Ensure we have a list of predictions
                if not isinstance(predictions, list):
                    predictions = [predictions]
                    
                # Ensure each prediction has the required fields
                for pred in predictions:
                    if 'class' not in pred:
                        pred['class'] = 'Unknown'
                    if 'confidence' not in pred:
                        pred['confidence'] = 0.0
                
                # Sort predictions by confidence (highest first) and limit to top_k
                predictions = sorted(predictions, 
                                  key=lambda x: float(x.get('confidence', 0)), 
                                  reverse=True)[:top_k]
                
                return predictions
                
            except (json.JSONDecodeError, AttributeError) as e:
                # If JSON parsing fails, return a simple response with the model's text
                print(f"Warning: Could not parse response as JSON: {e}")
                print(f"Response was: {response.text}")
                
                return [{
                    'class': 'Parsing error - See raw response',
                    'confidence': 0.0,
                    'raw_response': response.text
                }]
                
        except Exception as e:
            print(f"Error in Gemini prediction: {str(e)}")
            return [{
                'class': 'Error processing image',
                'confidence': 0.0,
                'error': str(e)
            }]
