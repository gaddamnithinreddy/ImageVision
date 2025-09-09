#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r image_recognition_project/requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example"
    cp .env.example .env
    echo "Please edit the .env file and add your GEMINI_API_KEY"
    exit 1
fi

# Load environment variables
set -a
source .env
set +a

# Create necessary directories
mkdir -p static/uploads

# Run the application
python -m image_recognition_project.app
