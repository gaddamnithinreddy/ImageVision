import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64
import json
import logging
from dotenv import load_dotenv
from models import GeminiModel
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['MAX_CONTENT_PATH'] = 16 * 1024 * 1024  # 16MB max file size

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'error': 'File too large',
        'message': 'The file is larger than the 16MB limit. Please choose a smaller file.'
    }), 413

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the model
model = None
api_key = os.getenv('GEMINI_API_KEY')

if not api_key:
    logger.error("GEMINI_API_KEY not found in environment variables")
    logger.info("Please set the GEMINI_API_KEY environment variable")
    logger.info("See .env.example for reference")
else:
    try:
        model = GeminiModel(api_key=api_key)
        logger.info("Successfully initialized Gemini model")
    except Exception as e:
        logger.error(f"Error initializing Gemini model: {str(e)}")
    model = None

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    logger = logging.getLogger(__name__)
    try:
        logger.info("\n=== New Recognition Request ===")
        logger.info(f"Request content length: {request.content_length} bytes")
        logger.info(f"Request files: {request.files}")
        
        if model is None:
            error_msg = 'Gemini model not properly initialized. Please check your API key.'
            print(f"Error: {error_msg}")
            return jsonify({'error': error_msg}), 500
            
        if 'file' not in request.files:
            error_msg = 'No file part in the request.'
            print(f"Error: {error_msg}")
            return jsonify({'error': error_msg}), 400
        
        file = request.files['file']
        print(f"Received file: {file.filename} ({file.content_length} bytes)")
        
        if file.filename == '':
            error_msg = 'No file selected.'
            print(f"Error: {error_msg}")
            return jsonify({'error': error_msg}), 400
        
        if not file:
            error_msg = 'No file data received.'
            print(f"Error: {error_msg}")
            return jsonify({'error': error_msg}), 400
            
        if not allowed_file(file.filename):
            error_msg = f'Invalid file type. Supported formats: {ALLOWED_EXTENSIONS}'
            print(f"Error: {error_msg}")
            return jsonify({'error': error_msg}), 400
        
        try:
            # Read and verify image file
            img_bytes = file.read()
            if not img_bytes:
                raise ValueError("Received empty file")
                
            print(f"Read {len(img_bytes)} bytes of image data")
            
            # Try to open the image to verify it's valid
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                print(f"Successfully loaded image: {img.width}x{img.height} pixels")
            except Exception as img_error:
                raise ValueError(f"Invalid image file: {str(img_error)}")
            
            # Preprocess image for Gemini
            print("Preprocessing image...")
            img_data = model.preprocess_image(img)
            
            # Make prediction with timeout
            print("Sending image to Gemini model...")
            try:
                logger.info("Sending request to Gemini model...")
                predictions = model.predict(img_data, top_k=5, timeout_seconds=30)
                logger.info(f"Received predictions: {predictions}")
                
                # Validate predictions
                if not isinstance(predictions, list):
                    raise ValueError(f"Unexpected response format: {type(predictions).__name__}")
                    
                if not predictions:
                    logger.warning("Received empty predictions list from model")
            except Exception as e:
                error_msg = str(e)
                if "timed out" in error_msg.lower():
                    raise Exception("The request to the AI model timed out. Please try again with a smaller image or check your internet connection.")
                elif "blocked" in error_msg.lower():
                    raise Exception("This image was blocked by content safety filters. Please try a different image.")
                else:
                    raise Exception(f"Error processing image: {error_msg}")
            
            # Convert image to base64 for display
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Process and validate predictions
            processed_predictions = []
            for i, pred in enumerate(predictions, 1):
                if not isinstance(pred, dict):
                    print(f"Warning: Prediction {i} is not a dictionary: {pred}")
                    continue
                    
                pred_class = str(pred.get('class', f'Unknown {i}')).strip()
                try:
                    confidence = float(pred.get('confidence', 0))
                    confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
                except (ValueError, TypeError) as e:
                    print(f"Warning: Invalid confidence value in prediction {i}: {pred.get('confidence')}")
                    confidence = 0.0
                
                processed_predictions.append({
                    'class': pred_class,
                    'confidence': confidence
                })
            
            if not processed_predictions:
                processed_predictions.append({
                    'class': 'No objects detected',
                    'confidence': 0.0
                })
                
            print(f"Returning {len(processed_predictions)} predictions")
            return jsonify({
                'image': f"data:image/jpeg;base64,{img_str}",
                'predictions': processed_predictions
            })
            
        except Exception as e:
            error_msg = f'Error processing image: {str(e)}'
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'error': 'Error processing image',
                'details': str(e),
                'type': type(e).__name__
            }), 500
            
    except Exception as e:
        error_msg = f'Unexpected error in /recognize endpoint: {str(e)}'
        print(f"CRITICAL ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Internal server error',
            'details': str(e),
            'type': type(e).__name__
        }), 500

@app.route('/generate-caption', methods=['POST'])
def generate_caption():
    try:
        data = request.get_json()
        image_data = data.get('image_data')
        platform = data.get('platform', 'general')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
            
        # Prepare the prompt based on the platform
        prompts = {
            'instagram': "Generate a single, clean Instagram caption for this image. Include 3-5 relevant hashtags and 1-2 emojis. Return only the caption text, no markdown, no options, no explanations. Example: 'Enjoying the sunshine! ☀️ #summer #happy #outdoors'",
            'twitter': "Create a single, clean Twitter caption for this image. Keep it under 280 characters with 1-3 relevant hashtags. Return only the caption text, no markdown, no options. Example: 'Beautiful day for a walk in the park! #nature #outdoors'",
            'linkedin': "Generate a single, professional LinkedIn caption for this image. Keep it concise and engaging. Return only the caption text, no markdown, no options. Example: 'Excited to share this moment from our team outing! #networking #teambuilding'",
            'whatsapp': "Create a single, casual WhatsApp caption for this image. Keep it short and friendly. Return only the caption text, no markdown, no options. Example: 'Guess where I am! 😊'",
            'general': "Generate a single, clean social media caption for this image. Keep it engaging and concise. Return only the caption text, no markdown, no options."
        }
        
        prompt = prompts.get(platform.lower(), prompts['general'])
        
        # Generate caption using Gemini
        response = model.model.generate_content([
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": image_data.split(',')[1] if ',' in image_data else image_data}}
        ])
        
        caption = response.text.strip()
        
        # Clean up the caption
        def clean_caption(text):
            # Remove markdown formatting (**bold**, _italic_, etc.)
            text = text.replace('**', '').replace('__', '').replace('*', '').replace('_', '')
            
            # Remove any numbered options (e.g., "1. Caption")
            import re
            text = re.sub(r'^\s*\d+[.)]?\s*', '', text, flags=re.MULTILINE)
            
            # Remove any section headers (e.g., "Option 1:", "Caption:")
            text = re.sub(r'^(option\s*\d+|caption|suggestion):?\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
            
            # Remove any remaining markdown links or special characters
            text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # [text](url) -> text
            
            # Take only the first line if there are multiple options
            text = text.split('\n')[0].strip()
            
            # Ensure proper spacing around emojis
            text = re.sub(r'([^\s])(:)', r'\1 :', text)  # Add space before emoji
            text = re.sub(r'(:)([^\s])', r'\1 \2', text)  # Add space after emoji
            
            return text.strip()
        
        # Clean the caption
        clean_caption_text = clean_caption(caption)
        
        return jsonify({
            'success': True,
            'caption': clean_caption_text
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error generating caption: {str(e)}'
        }), 500

def create_app():
    # Ensure the upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    return app

if __name__ == '__main__':
    # Create the app
    app = create_app()
    
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    logger.info(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
