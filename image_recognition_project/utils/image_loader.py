import os
import requests
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from typing import Union, Tuple, Optional

def is_valid_url(url: str) -> bool:
    """Check if the given string is a valid URL."""
    import re
    regex = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?'  # domain...
        r'|localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None

def download_image(url: str, timeout: int = 10) -> Image.Image:
    """
    Download an image from a URL and return it as a PIL Image.
    
    Args:
        url (str): URL of the image to download
        timeout (int): Timeout in seconds for the request
        
    Returns:
        PIL.Image: The downloaded image
        
    Raises:
        ValueError: If the URL is invalid or the image cannot be downloaded
    """
    if not is_valid_url(url):
        raise ValueError(f"Invalid URL: {url}")
    
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        # Check if the response contains an image
        if 'image' not in response.headers.get('content-type', '').lower():
            raise ValueError("URL does not point to a valid image")
            
        return Image.open(BytesIO(response.content)).convert('RGB')
    except requests.RequestException as e:
        raise ValueError(f"Failed to download image from {url}: {str(e)}")
    except UnidentifiedImageError:
        raise ValueError("Downloaded content is not a valid image")

def load_image(image_path: str) -> Image.Image:
    """
    Load an image from a file path or URL.
    
    Args:
        image_path (str): Path to the image file or URL
        
    Returns:
        PIL.Image: The loaded image
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is not a valid image or URL is invalid
    """
    if is_valid_url(image_path):
        return download_image(image_path)
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    try:
        return Image.open(image_path).convert('RGB')
    except UnidentifiedImageError:
        raise ValueError(f"File is not a valid image: {image_path}")
