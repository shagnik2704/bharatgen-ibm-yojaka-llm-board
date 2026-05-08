"""
Extract raw PDF text using PyMuPDF/EasyOCR without any cleaning.
Saves raw output to JSON for inspection and citation extraction analysis.
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

try:
    from .pdf_loader import load_pdf
except ImportError:
    from utils.pdf_loader import load_pdf


def extract_raw_from_folder(input_dir: str, output_file: str = None):
    """
    Extract raw text from all PDFs in a folder using PyMuPDF/EasyOCR.
    Save uncleaned output to JSON with minimal processing.
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Input path not found: {input_dir}")
        sys.exit(1)
    
    # Auto-name output if not provided
    if not output_file:
        output_file = f"raw_extract_{input_path.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    output_path = Path(output_file)
    
    # Find all PDFs
    if input_path.is_file():
        pdfs = [input_path]
        base_input = input_path.parent
    else:
        pdfs = sorted(input_path.rglob("*.pdf"))
        base_input = input_path
    
    print(f"Found {len(pdfs)} PDFs in {input_path}")
    print(f"Output will be saved to: {output_path.absolute()}\n")
    
    all_extracts = {}
    
    for i, pdf_path in enumerate(pdfs, 1):
        rel_path = pdf_path.relative_to(base_input)
        print(f"[{i}/{len(pdfs)}] Extracting: {rel_path}")
        
        try:
            # Use the same load_pdf as process_pdfs.py (with PyMuPDF + EasyOCR)
            pages, full_text = load_pdf(str(pdf_path), force_ocr=False)
            
            # Store raw output with metadata
            extract_record = {
                "pdf_file": pdf_path.name,
                "relative_path": str(rel_path),
                "absolute_path": str(pdf_path),
                "total_pages": len(pages),
                "page_texts": pages,  # Raw pages before concatenation
                "full_text": full_text,  # Concatenated raw text
                "full_text_length": len(full_text),
                "extraction_timestamp": datetime.now().isoformat(),
            }
            
            all_extracts[pdf_path.name] = extract_record
            print(f"  ✓ Extracted {len(pages)} pages, {len(full_text):,} characters")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            all_extracts[pdf_path.name] = {
                "error": str(e),
                "pdf_file": pdf_path.name,
                "relative_path": str(rel_path),
            }
    
    # Save to JSON
    output_dict = {
        "extraction_summary": {
            "timestamp": datetime.now().isoformat(),
            "input_folder": str(input_path.absolute()),
            "total_pdfs_processed": len(pdfs),
            "successful_extracts": sum(1 for v in all_extracts.values() if "error" not in v),
            "failed_extracts": sum(1 for v in all_extracts.values() if "error" in v),
        },
        "extracts": all_extracts,
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Raw extraction saved to: {output_path.absolute()}")
    print(f"  Total successful: {output_dict['extraction_summary']['successful_extracts']}")
    print(f"  Total failed: {output_dict['extraction_summary']['failed_extracts']}")
    
    return str(output_path.absolute())


def main():
    parser = argparse.ArgumentParser(
        description="Extract raw PDF text using PyMuPDF/EasyOCR and save to JSON."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input PDF file or folder containing PDFs"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON file (defaults to raw_extract_<folder>_<timestamp>.json)"
    )
    args = parser.parse_args()
    
    extract_raw_from_folder(args.input, args.output)


if __name__ == "__main__":
    main()
