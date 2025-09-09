import os
import google.generativeai as genai
from PIL import Image
import io
import base64
import json

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
    
    def _generate_content_with_timeout(self, model, contents, generation_config=None, **kwargs):
        """Helper method to run generate_content with a timeout using threading."""
        from queue import Queue, Empty
        import threading
        
        result_queue = Queue()
        
        def worker():
            try:
                # The timeout kwarg is for the queue, not the API call itself.
                api_kwargs = kwargs.copy()
                api_kwargs.pop('timeout', None)
                response = model.generate_content(contents, generation_config=generation_config, **api_kwargs)
                result_queue.put(('success', response))
            except Exception as e:
                result_queue.put(('error', str(e)))
        
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        
        try:
            status, result = result_queue.get(timeout=kwargs.get('timeout', 30))
            if status == 'error':
                raise Exception(result)
            return result
        except Empty:
            raise TimeoutError("Request timed out. The server is taking too long to respond.")
        except Exception as e:
            if 'resource_exhausted' in str(e).lower() or 'rate limit' in str(e).lower():
                raise Exception("The image recognition service is currently busy. Please try again in a few minutes.")
            raise e
        finally:
            thread.join(timeout=0.1)  # Give thread a short time to clean up
    
    def predict(self, image_data, prompt=None, top_k=5, timeout_seconds=30):
        """
        Make a prediction using the Gemini model.
        
        Args:
            image_data: Image data from preprocess_image
            prompt (str): Optional custom prompt
            top_k (int): Number of predictions to return (must be positive)
            timeout_seconds (int): Maximum time to wait for the API response
            
        Returns:
            list: List of predictions with confidence scores
        """
        # Validate inputs
        if not isinstance(top_k, int) or top_k <= 0:
            top_k = 5  # Default to 5 if invalid
            
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            timeout_seconds = 30  # Default to 30 seconds if invalid
            
        if prompt is None:
            prompt = f"""
            Analyze this image and provide the top {top_k} most likely objects or scenes in the image.
            For each prediction, provide:
            1. The name of the object/scene
            2. A confidence score between 0 and 1
            
            Format the response as a valid JSON array of objects with 'class' and 'confidence' keys.
            Example response:
            [
                {{"class": "dog", "confidence": 0.95}},
                {{"class": "cat", "confidence": 0.85}}
            ]
            
            Important: Only return the JSON array, no other text or markdown formatting.
            """
        
        try:
            # Convert image data to bytes and validate
            try:
                img_bytes = base64.b64decode(image_data['data'])
                # Verify the image can be opened and get its size
                img = Image.open(io.BytesIO(img_bytes))
                img_size_mb = len(img_bytes) / (1024 * 1024)
                if img_size_mb > 4:  # Warn about large images
                    print(f"Warning: Large image detected ({img_size_mb:.2f} MB). Processing may be slow.")
            except Exception as img_error:
                raise ValueError(f"Invalid image data: {str(img_error)}")
            
            try:
                # Generate content with the image and prompt using our timeout wrapper
                print(f"Sending request to Gemini API (timeout: {timeout_seconds}s)...")
                response = self._generate_content_with_timeout(
                    self.model,
                    [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_data['data']}}
                    ],
                    timeout=timeout_seconds
                )
                
                if not hasattr(response, 'text') or not response.text:
                    error_detail = "No response text from API"
                    if hasattr(response, 'prompt_feedback'):
                        if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                            error_detail = f"Content blocked: {response.prompt_feedback.block_reason}"
                        if hasattr(response.prompt_feedback, 'safety_ratings'):
                            for rating in response.prompt_feedback.safety_ratings:
                                if rating.blocked:
                                    error_detail += f"\nBlocked by {rating.category} filter (confidence: {rating.probability})"
                    print(f"Error details: {error_detail}")
                    raise ValueError(error_detail)
                    
                response_text = response.text.strip()
                
                # Clean up the response text to make it valid JSON
                if not response_text:
                    raise ValueError("Empty response from API")
                    
                # Try to extract JSON from markdown code blocks if present
                if '```json' in response_text:
                    response_text = response_text.split('```json')[1].split('```')[0].strip()
                elif '```' in response_text:
                    response_text = response_text.split('```')[1].strip()
                
                # Parse the JSON response
                try:
                    predictions = json.loads(response_text)
                except json.JSONDecodeError as je:
                    # Try to find JSON array in the response
                    import re
                    json_match = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL)
                    if json_match:
                        try:
                            predictions = json.loads(json_match.group(0))
                        except json.JSONDecodeError:
                            raise ValueError(f"Could not parse JSON from response: {response_text[:200]}...")
                    else:
                        raise ValueError(f"No valid JSON found in response: {response_text[:200]}...")
                
                # Ensure we have a list of predictions
                if not isinstance(predictions, list):
                    predictions = [predictions]
                
                # Process predictions
                processed_predictions = []
                for i, pred in enumerate(predictions, 1):
                    if not isinstance(pred, dict):
                        continue
                        
                    # Extract class and confidence with validation
                    pred_class = str(pred.get('class', f'Unknown {i}')).strip()
                    try:
                        confidence = float(pred.get('confidence', 0))
                        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
                    except (ValueError, TypeError):
                        confidence = 0.0
                    
                    processed_predictions.append({
                        'class': pred_class,
                        'confidence': confidence
                    })
                
                # If no valid predictions, add a default one
                if not processed_predictions:
                    processed_predictions.append({
                        'class': 'No objects detected',
                        'confidence': 0.0
                    })
                
                # Sort predictions by confidence (highest first) and limit to top_k
                processed_predictions.sort(key=lambda x: x['confidence'], reverse=True)
                return processed_predictions[:top_k]
                
            except Exception as api_error:
                error_msg = str(api_error)
                if "safety" in error_msg.lower():
                    return [{
                        'class': 'Content blocked by safety settings',
                        'confidence': 0.0,
                        'error': 'The image was blocked by content safety filters.'
                    }]
                raise  # Re-raise other API errors
                
        except Exception as e:
            print(f"Error in Gemini prediction: {str(e)}")
            return [{
                'class': 'Error processing image',
                'confidence': 0.0,
                'error': str(e)
            }]
