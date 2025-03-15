import numpy as np
from PIL import Image

def preprocess_image(pil_image):
    """Preprocess image to improve OCR accuracy"""
    try:
        # Convert to grayscale
        img_gray = pil_image.convert('L')
        
        # Simple thresholding to enhance text
        threshold = 200
        img_bw = img_gray.point(lambda x: 0 if x < threshold else 255, '1')
        
        return img_bw
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return pil_image  # Return original image if processing fails