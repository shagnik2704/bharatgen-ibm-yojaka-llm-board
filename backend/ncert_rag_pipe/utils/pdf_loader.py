"""
PDF Loader - Generalized Clean Text Extraction with Smart OCR (EasyOCR)
=====================================================================
Extracts raw text from PDFs. Automatically detects scanned images 
AND digitally typed PDFs using broken legacy Hindi fonts (Kruti Dev, etc.),
falling back to EasyOCR when necessary for high-accuracy Hindi/English extraction.
"""

import logging
import re
from pathlib import Path
from typing import Tuple, List

import pymupdf as fitz  # PyMUPDF
import numpy as np
try:
    import easyocr
except ImportError:
    easyocr = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize EasyOCR reader once globally to save massive model loading time.
# If easyocr is not installed, OCR paths are disabled and extractor will use native PDF text only.
reader = None
if easyocr is not None:
    logger.info("Initializing EasyOCR for Hindi and English...")
    reader = easyocr.Reader(['hi', 'en'])
else:
    logger.warning("easyocr not installed. OCR fallback is disabled.")

def is_garbage_text(text: str, threshold: float = 0.10) -> bool:
    """
    Detects if the extracted text is gibberish/legacy font mapping.
    Legacy Hindi fonts map Devanagari to weird extended ASCII characters.
    """
    if len(text) < 50:
        return True # Too short, likely an image-based page
        
    # Find all characters that are NOT:
    # 1. Standard English/ASCII ( \x20-\x7E )
    # 2. Standard Whitespace ( \s )
    # 3. Unicode Devanagari / Hindi ( \u0900-\u097F )
    weird_chars = re.findall(r'[^\x20-\x7E\s\u0900-\u097F]', text)
    
    # If the percentage of "weird" characters is higher than the threshold,
    # it means the PDF is using a broken legacy font encoding.
    weird_ratio = len(weird_chars) / len(text)
    
    if weird_ratio > threshold:
        logger.warning(f"Legacy font/gibberish detected! (Weird char ratio: {weird_ratio:.1%})")
        return True
        
    return False

def clean_page_text(text: str) -> str:
    """Removes general PDF visual noise (headers, footers, page numbers)."""
    if not text:
        return ""
        
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\.{10,}', '', text)
    text = re.sub(r'(?i)^\s*(Reprint\s*\d{4}-\d{2}|not to be republished)\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*([A-Z][a-zA-Z\s&]+){3,}\s*$', '', text, flags=re.MULTILINE)
    
    return text.strip()

def extract_text_with_ocr(page: fitz.Page) -> str:
    """Renders a PDF page to a high-res numpy array and applies EasyOCR."""
    if reader is None:
        return ""
    try:
        # Render at 200 DPI (Excellent balance of speed and accuracy for Deep Learning OCR)
        pix = page.get_pixmap(matrix=fitz.Matrix(200/72, 200/72))
        
        # Convert PyMuPDF pixmap to numpy array directly for EasyOCR
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        # If the image has an alpha channel (RGBA), drop it to make it RGB
        if pix.alpha:
            img_array = img_array[:, :, :3]

        # Extract text:
        # detail=0 returns just the text strings
        # paragraph=True intelligently merges bounding boxes into proper paragraphs
        result = reader.readtext(img_array, detail=0, paragraph=True)
        text = "\n\n".join(result)
        
        return text
    except Exception as e:
        logger.error(f"OCR failed on page: {e}")
        return ""

def load_pdf(pdf_path: str, force_ocr: bool = False) -> Tuple[List[str], str]:
    """Loads a PDF with smart OCR fallback for images and legacy fonts."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    logger.info(f"Loading PDF: {pdf_path.name}")
    doc = fitz.open(str(pdf_path))
    pages = []
    
    ocr_triggered_count = 0
    
    for page_num, page in enumerate(doc):
        text = ""
        
        if not force_ocr:
            text = page.get_text("text").strip()
            
        # SMART DETECTION:
        # Triggers OCR if forced, or if `is_garbage_text` detects images / legacy fonts
        if force_ocr or is_garbage_text(text):
            if reader is not None:
                logger.info(f"Page {page_num + 1} requires OCR. Running EasyOCR (hi/en)...")
                text = extract_text_with_ocr(page)
                ocr_triggered_count += 1
            else:
                logger.warning(
                    f"Page {page_num + 1} appears OCR-needed, but easyocr is unavailable. "
                    "Using native extraction result only."
                )
            
        cleaned_text = clean_page_text(text)
        if cleaned_text:
            pages.append(cleaned_text)
        
    doc.close()
    
    full_text = "\n\n".join(pages)
    
    if ocr_triggered_count > 0:
        logger.info(f"✓ Applied EasyOCR to {ocr_triggered_count} pages (Scanned or Legacy Fonts).")
        
    logger.info(f"✓ Loaded {len(pages)} pages, {len(full_text):,} characters")
    
    return pages, full_text