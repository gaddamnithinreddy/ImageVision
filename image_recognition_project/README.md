# VisionAI - Image Recognition & Captioning

A web application that uses Google's Gemini AI to recognize objects in images and generate social media captions.

## Features

- Upload images for object recognition
- Generate captions for social media platforms (Instagram, Twitter, LinkedIn, WhatsApp)
- Clean and modern user interface
- Responsive design that works on desktop and mobile
- Top-k predictions with confidence scores
- GPU acceleration support (if available)
- Batch processing capabilities

## Requirements

- Python 3.7+
- PyTorch 1.9.0+
- TorchVision 0.10.0+
- Pillow 8.0.0+
- Other dependencies in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd image_recognition_project
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
python main.py path/to/your/image.jpg
```

### Options

- `--model`: Choose model (alexnet or resnet101, default: resnet101)
- `--topk`: Number of top predictions to show (default: 5)
- `--device`: Device to run on (cuda or cpu, default: auto-detect)
- `--output`: Path to save results as JSON

### Examples

1. Classify an image with default settings (ResNet101):
   ```bash
   python main.py example.jpg
   ```

2. Use AlexNet instead of ResNet101:
   ```bash
   python main.py --model alexnet example.jpg
   ```

3. Show top 3 predictions and save results to a file:
   ```bash
   python main.py --topk 3 --output results.json example.jpg
   ```

4. Classify an image from a URL:
   ```bash
   python main.py "https://example.com/image.jpg"
   ```

## Project Structure

```
image_recognition_project/
├── models/               # Model implementations
│   ├── __init__.py
│   ├── alexnet_model.py  # AlexNet implementation
│   └── resnet_model.py   # ResNet101 implementation
├── utils/                # Utility functions
│   ├── __init__.py
│   ├── image_loader.py   # Image loading utilities
│   └── preprocessing.py  # Image preprocessing
├── data/
│   └── imagenet_classes.txt  # ImageNet class labels
├── main.py              # Main script
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## Adding New Models

To add a new model:

1. Create a new model class in the `models` directory following the same pattern as the existing models
2. Add the model to the `MODEL_MAP` in `main.py`
3. The model class should implement:
   - `__init__`: Initialize the model with pre-trained weights
   - `preprocess_image`: Method to preprocess input images
   - `predict`: Method to make predictions

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch and TorchVision teams for the pre-trained models
- ImageNet dataset for the training data
