"""
Pure Streamlit implementation of the VisionAI Image Recognition application.
This version eliminates the need for Flask and runs entirely on Streamlit.
"""

import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import base64
import json
from PIL import Image
import io
import time

# Load environment variables
load_dotenv()

# Configure the API
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    st.error("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop()

genai.configure(api_key=api_key)

# Initialize the model
@st.cache_resource
def get_model():
    """Initialize and cache the Gemini model."""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        return model
    except Exception as e:
        st.error(f"Error initializing Gemini model: {str(e)}")
        return None

def preprocess_image(image):
    """
    Convert PIL Image to a format suitable for Gemini API.
    
    Args:
        image: PIL Image object
        
    Returns:
        dict: Processed image data
    """
    # Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    return {
        'mime_type': 'image/jpeg',
        'data': base64.b64encode(img_byte_arr).decode('utf-8')
    }

def predict(model, image_data, top_k=5):
    """
    Make a prediction using the Gemini model.
    
    Args:
        model: Gemini model instance
        image_data: Image data from preprocess_image
        top_k (int): Number of predictions to return
            
    Returns:
        list: List of predictions with confidence scores
    """
    if not isinstance(top_k, int) or top_k <= 0:
        top_k = 5  # Default to 5 if invalid
        
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
        # Generate content with the image and prompt
        response = model.generate_content([
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": image_data['data']}}
        ])
        
        if not hasattr(response, 'text') or not response.text:
            return [{
                'class': 'No response from AI model',
                'confidence': 0.0
            }]
            
        response_text = response.text.strip()
        
        # Clean up the response text to make it valid JSON
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].strip()
        
        # Parse the JSON response
        try:
            predictions = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to find JSON array in the response
            import re
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL)
            if json_match:
                try:
                    predictions = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    predictions = [{
                        'class': 'Error parsing AI response',
                        'confidence': 0.0,
                        'error': response_text[:100] + '...'
                    }]
            else:
                predictions = [{
                    'class': 'No valid JSON found in response',
                    'confidence': 0.0,
                    'error': response_text[:100] + '...'
                }]
        
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
        
    except Exception as e:
        return [{
            'class': f'Error processing image: {str(e)}',
            'confidence': 0.0
        }]

def generate_caption(model, image_data, platform='general'):
    """
    Generate a social media caption for the image.
    
    Args:
        model: Gemini model instance
        image_data: Image data
        platform (str): Social media platform
        
    Returns:
        str: Generated caption
    """
    # Prepare the prompt based on the platform
    prompts = {
        'instagram': "Generate a single, clean Instagram caption for this image. Include 3-5 relevant hashtags and 1-2 emojis. Return only the caption text, no markdown, no options, no explanations. Example: 'Enjoying the sunshine! ☀️ #summer #happy #outdoors'",
        'twitter': "Create a single, clean Twitter caption for this image. Keep it under 280 characters with 1-3 relevant hashtags. Return only the caption text, no markdown, no options. Example: 'Beautiful day for a walk in the park! #nature #outdoors'",
        'linkedin': "Generate a single, professional LinkedIn caption for this image. Keep it concise and engaging. Return only the caption text, no markdown, no options. Example: 'Excited to share this moment from our team outing! #networking #teambuilding'",
        'whatsapp': "Create a single, casual WhatsApp caption for this image. Keep it short and friendly. Return only the caption text, no markdown, no options. Example: 'Guess where I am! 😊'",
        'general': "Generate a single, clean social media caption for this image. Keep it engaging and concise. Return only the caption text, no markdown, no options."
    }
    
    prompt = prompts.get(platform.lower(), prompts['general'])
    
    try:
        # Generate caption using Gemini
        response = model.generate_content([
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": image_data['data']}}
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
        return clean_caption_text
        
    except Exception as e:
        return f"Error generating caption: {str(e)}"

def main():
    """Main Streamlit application."""
    # Set up the page
    st.set_page_config(
        page_title="VisionAI - Image Recognition",
        page_icon="👁️",
        layout="wide"
    )
    
    # Custom CSS to make it look more like the original HTML design
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .gradient-text {
            background: linear-gradient(90deg, #6366f1, #8b5cf6);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
        }
        .drop-zone {
            border: 2px dashed #9ca3af;
            border-radius: 1rem;
            padding: 2rem;
            text-align: center;
            margin-bottom: 2rem;
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(8px);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .drop-zone:hover {
            border-color: #6366f1;
            background: rgba(255, 255, 255, 0.9);
            transform: translateY(-2px);
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
        }
        .prediction-item {
            transition: all 0.3s ease;
            border-left: 4px solid transparent;
            padding: 1rem;
            margin-bottom: 0.5rem;
            border-radius: 0.5rem;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .prediction-item:hover {
            transform: translateX(5px);
            border-left-color: #6366f1;
        }
        .confidence-bar {
            height: 1.5rem;
            border-radius: 0.75rem;
            background: linear-gradient(to right, #6366f1, #8b5cf6);
            margin: 0.5rem 0;
            position: relative;
            overflow: hidden;
        }
        .confidence-text {
            position: absolute;
            right: 0.5rem;
            top: 50%;
            transform: translateY(-50%);
            color: white;
            font-weight: bold;
            font-size: 0.8rem;
        }
        .btn {
            transition: all 0.2s ease;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        .btn:hover {
            transform: translateY(-1px);
        }
        .btn:active {
            transform: translateY(0);
        }
        .card {
            background: white;
            border-radius: 1rem;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        .platform-btn {
            margin: 0.25rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title and description with gradient text
    st.markdown("""
        <div class="main-header">
            <h1><span class="gradient-text">VisionAI</span></h1>
            <p>Upload an image and let our AI analyze its contents with advanced computer vision technology.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize the model
    model = get_model()
    if model is None:
        st.error("Failed to initialize the AI model. Please check your API key and try again.")
        return
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Image Recognition", "How It Works"])
    
    with tab1:
        # File uploader with custom styling
        st.markdown('<div class="drop-zone">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drag & drop your image here or click to browse files",
            type=["png", "jpg", "jpeg", "gif"],
            label_visibility="collapsed",
            help="Supports PNG, JPG, JPEG, and GIF files (Max 16MB)"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Display the uploaded image
            try:
                image = Image.open(uploaded_file).convert('RGB')
                
                # Preprocess the image
                with st.spinner("Processing your image..."):
                    img_data = preprocess_image(image)
                    
                    # Make prediction
                    with st.spinner("Analyzing image with AI... This usually takes a few seconds"):
                        predictions = predict(model, img_data, top_k=5)
                
                # Display results in a card layout
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Analysis Results")
                
                # Create two columns for image preview and predictions
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, caption="Processed Image", use_container_width=True)
                
                with col2:
                    st.subheader("Confidence Scores")
                    
                    # Display predictions with styled progress bars
                    for i, prediction in enumerate(predictions):
                        if 'class' in prediction and 'confidence' in prediction:
                            confidence_percent = min(round(prediction['confidence'] * 100), 100)
                            
                            # Create a styled prediction item
                            st.markdown(f"""
                                <div class="prediction-item">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <span style="font-weight: 500;">{i+1}. {prediction['class']}</span>
                                        <span style="font-size: 0.9rem; color: #6b7280;">{confidence_percent}% confidence</span>
                                    </div>
                                    <div class="confidence-bar">
                                        <div style="width: {prediction['confidence']*100}%; height: 100%; background: linear-gradient(to right, #6366f1, #8b5cf6); border-radius: 0.75rem;"></div>
                                        <div class="confidence-text">{confidence_percent}%</div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Social Media Caption Generator in a card
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Generate Social Media Caption")
                
                # Platform selection buttons
                platforms = ["Instagram", "Twitter", "LinkedIn", "WhatsApp", "General"]
                platform_cols = st.columns(len(platforms))
                
                selected_platform = None
                for i, platform in enumerate(platforms):
                    if platform_cols[i].button(platform, key=f"platform_{platform.lower()}", use_container_width=True):
                        selected_platform = platform
                
                # Generate caption if a platform is selected
                if selected_platform:
                    with st.spinner("Generating caption..."):
                        caption = generate_caption(model, img_data, selected_platform.lower())
                        
                        # Display the generated caption in a styled box
                        st.markdown(f"""
                            <div style="background: #f3f4f6; border-radius: 0.5rem; padding: 1rem; margin-top: 1rem;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                    <span style="font-weight: 500;">Suggested Caption for {selected_platform}:</span>
                                </div>
                                <div style="white-space: pre-wrap; font-size: 0.9rem;">{caption}</div>
                                <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #6b7280;">
                                    Copy the caption above manually (Ctrl+C / Cmd+C)
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
        else:
            # Show example images or instructions
            st.info("👆 Please upload an image to get started!")
            st.markdown("""
            ### Supported Formats:
            - PNG
            - JPG/JPEG
            - GIF
            
            ### Tips for Best Results:
            - Use clear, well-lit images
            - Avoid blurry or heavily compressed images
            - Images with recognizable objects work best
            """)
    
    with tab2:
        st.subheader("How VisionAI Works")
        st.markdown("""
        ### Technology Stack
        - **AI Model**: Google's Gemini 2.0 Flash for image recognition
        - **Framework**: Streamlit for the web interface
        - **Image Processing**: PIL (Python Imaging Library)
        
        ### Process Flow
        1. **Image Upload**: You upload an image file
        2. **Preprocessing**: Image is converted to the proper format
        3. **AI Analysis**: Gemini model analyzes the image content
        4. **Prediction**: Top 5 recognized objects with confidence scores
        5. **Caption Generation**: Optional social media captions
        
        ### Privacy & Security
        - Images are processed in real-time and not stored
        - Your API key is securely loaded from environment variables
        - All processing happens through Google's secure AI APIs
        """)
        
        st.subheader("About the AI Model")
        st.markdown("""
        VisionAI uses Google's Gemini 2.0 Flash model, which is optimized for:
        - Fast image recognition
        - High accuracy object detection
        - Natural language understanding
        - Multi-modal capabilities (text and image processing)
        
        The model can recognize thousands of objects, scenes, and concepts in images.
        """)

if __name__ == "__main__":
    main()