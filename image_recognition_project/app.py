from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64
import json
from models import GeminiModel
import google.generativeai as genai

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the model
try:
    # Try to get API key from environment variable
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        # If not found, try to read from a file
        try:
            with open('gemini_api_key.txt', 'r') as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            api_key = 'YOUR_API_KEY_HERE'  # Will prompt user to set it
    
    model = GeminiModel(api_key=api_key)
except Exception as e:
    print(f"Error initializing Gemini model: {str(e)}")
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
    if model is None:
        return jsonify({'error': 'Gemini model not properly initialized. Please check your API key.'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Read image file
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            
            # Preprocess image for Gemini
            img_data = model.preprocess_image(img)
            
            # Make prediction
            predictions = model.predict(img_data, top_k=5)
            
            # Convert image to base64 for display
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Ensure confidence is between 0 and 1
            processed_predictions = []
            for pred in predictions:
                confidence = float(pred.get('confidence', 0))
                # Ensure confidence is between 0 and 1
                confidence = max(0.0, min(1.0, confidence))
                processed_predictions.append({
                    'class': pred.get('class', 'Unknown'),
                    'confidence': confidence
                })
                
            return jsonify({
                'image': f"data:image/jpeg;base64,{img_str}",
                'predictions': processed_predictions
            })
            
        except Exception as e:
            return jsonify({
                'error': f'Error processing image: {str(e)}',
                'details': str(e)
            }), 500
    
    return jsonify({'error': 'Invalid file type. Supported formats: PNG, JPG, JPEG, GIF'}), 400

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

if __name__ == '__main__':
    app.run(debug=True, port=5001)
