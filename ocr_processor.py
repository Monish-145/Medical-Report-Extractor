"""
Enhanced OCR and text extraction module with advanced image processing and multiple OCR engines
"""
import os
import re
from typing import List, Tuple, Dict, Any

# Optional imports with fallbacks
try:
    import cv2
    import numpy as np
    from PIL import Image
    HAS_IMAGE_PROCESSING = True
except ImportError:
    HAS_IMAGE_PROCESSING = False

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF using basic methods
    """
    try:
        # Try to use PyPDF2 first (lightweight)
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except ImportError:
        pass
    
    try:
        # Try pdfplumber as backup
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except ImportError:
        pass
    
    try:
        # Try pymupdf as another option
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text.strip()
    except ImportError:
        pass
    
    # If no PDF library is available, return a message
    return f"Could not extract text from PDF: {os.path.basename(pdf_path)}. Please install PyPDF2, pdfplumber, or pymupdf."

def preprocess_image_for_ocr(image_path: str) -> str:
    """
    Advanced image preprocessing for better OCR results
    """
    if not HAS_IMAGE_PROCESSING:
        return image_path
        
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Image not found at {image_path}")
            return image_path
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        processed = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
        
        # Save processed image
        processed_path = f"processed_{os.path.basename(image_path)}"
        os.makedirs(os.path.dirname(processed_path) if os.path.dirname(processed_path) else ".", exist_ok=True)
        cv2.imwrite(processed_path, processed)
        return processed_path
        
    except Exception as e:
        print(f"Image preprocessing failed: {e}")
        return image_path

def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from image using multiple OCR engines with preprocessing
    """
    # Preprocess the image first (if image processing is available)
    processed_path = preprocess_image_for_ocr(image_path)
    
    # Try multiple OCR engines
    ocr_results = []
    
    # 1. Try pytesseract with different configurations
    try:
        import pytesseract
        from PIL import Image
        
        image = Image.open(processed_path)
        
        # Try different OCR configurations
        configs = [
            '--psm 6',  # Uniform block of text
            '--psm 4',  # Single column of text
            '--psm 3',  # Fully automatic page segmentation
            '--psm 1',  # Automatic page segmentation with OSD
        ]
        
        for config in configs:
            try:
                text = pytesseract.image_to_string(image, config=config)
                if text.strip():
                    ocr_results.append(('tesseract', text.strip(), 0.8))
                    break  # Use first successful result
            except Exception as e:
                print(f"Tesseract config {config} failed: {e}")
                continue
                
    except ImportError:
        print("pytesseract not available")
    except Exception as e:
        print(f"pytesseract error: {e}")
    
    # 2. Try easyocr
    try:
        import easyocr
        reader = easyocr.Reader(['en'])
        results = reader.readtext(processed_path)
        text = " ".join([result[1] for result in results])
        if text.strip():
            ocr_results.append(('easyocr', text.strip(), 0.9))
    except ImportError:
        pass
    except Exception:
        pass
    
    # 3. Try doctr (if available)
    try:
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor
        
        # Load document
        doc = DocumentFile.from_images(processed_path)
        
        # Load model
        model = ocr_predictor(pretrained=True)
        
        # Run OCR
        result = model(doc)
        
        # Extract text
        text_parts = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    text_parts.append(line.value)
        
        text = " ".join(text_parts)
        if text.strip():
            ocr_results.append(('doctr', text.strip(), 0.95))
            
    except ImportError:
        pass
    except Exception:
        pass
    
    # Clean up processed image
    if processed_path != image_path and os.path.exists(processed_path):
        try:
            os.remove(processed_path)
        except:
            pass
    
    # Return the best result
    if ocr_results:
        # Sort by confidence and return the best
        ocr_results.sort(key=lambda x: x[2], reverse=True)
        return ocr_results[0][1]
    
    # Fallback: Try to extract text using basic image processing
    try:
        if HAS_IMAGE_PROCESSING:
            import cv2
            import numpy as np
            from PIL import Image
            
            # Load and process image
            img = cv2.imread(image_path)
            if img is not None:
                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Apply threshold
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Try to extract text using basic character recognition
                # This is a very basic fallback
                return f"Image processed but OCR engines not available. File: {os.path.basename(image_path)}"
    except Exception as e:
        print(f"Fallback processing failed: {e}")
    
    return f"Could not extract text from image: {os.path.basename(image_path)}. Please install pytesseract, easyocr, or doctr."

def simple_ocr(file_path: str) -> Tuple[List[Dict], str]:
    """
    Extract text from file (PDF or image) and return tokens and full text
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.pdf':
        full_text = extract_text_from_pdf(file_path)
    elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
        full_text = extract_text_from_image(file_path)
    else:
        full_text = f"Unsupported file format: {file_ext}"
    
    # Create simple tokens (mock bounding boxes for now)
    tokens = []
    words = full_text.split()
    for i, word in enumerate(words):
        # Create mock bounding box
        bbox = [i * 0.1, 0.1, (i + 1) * 0.1, 0.2]  # Normalized coordinates
        tokens.append({
            "text": word,
            "bbox": bbox,
            "confidence": 0.9
        })
    
    return tokens, full_text

# Install required packages function
def install_pdf_requirements():
    """
    Install required packages for PDF processing
    """
    import subprocess
    import sys
    
    packages = ['PyPDF2', 'pdfplumber', 'pytesseract', 'Pillow']
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        except subprocess.CalledProcessError:
            pass

if __name__ == "__main__":
    pass
