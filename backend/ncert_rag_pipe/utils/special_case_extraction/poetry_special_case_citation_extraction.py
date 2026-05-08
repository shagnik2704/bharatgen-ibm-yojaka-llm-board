import json
import re
import argparse
from pathlib import Path

def get_word_limit_offset(text: str, limit: int = 210) -> tuple:
    """
    Returns the text truncated to `limit` words and the character length 
    of the truncated text so we know exactly where to resume searching.
    """
    word_iter = re.finditer(r'\S+', text)
    count = 0
    last_end = len(text)
    for w in word_iter:
        count += 1
        last_end = w.end()
        if count == limit:
            break
    return text[:last_end].strip(), last_end

def process_special_citations(raw_text_path: Path, block_folder_path: Path):
    # Load the raw extracted text JSON
    with open(raw_text_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    full_text = data.get("full_text", "")
    source_pdf = data.get("source_pdf", "Unknown.pdf")
    block_label = block_folder_path.name
    
    # regex matches:
    # 1. 1 word, a colon, and AT LEAST ONE newline
    # 2. OR a sequence of numbers separated by periods, a newline, 'text', and a newline,
    #    PROVIDED it is NOT immediately followed by another number sequence.
    start_pattern = re.compile(
        r'(?:^|\n+)('
        r'[a-zA-Z]+:[ \t]*\n+'
        r'|'
        r'\d+(?:\.\d+)+[ \t]*\n+[ \t]*text[ \t]*\n+(?!\d+(?:\.\d+)+)'
        r')', re.IGNORECASE
    )
    
    # regex matches: The whole word "glossary"
    end_pattern = re.compile(r'\bglossary\b', re.IGNORECASE)

    # Find all potential start markers
    start_matches = list(start_pattern.finditer(full_text))
    
    extracted_citations = []
    idx = 0
    
    while idx < len(start_matches):
        match = start_matches[idx]
        
        # Extract the raw marker and clean up newlines/spaces for the JSON metadata
        marker_raw = match.group(1)
        marker_clean = re.sub(r'\s+', ' ', marker_raw).strip()
        
        # The start of our block includes the start marker
        start_pos = match.start()
        
        # Look for the ending "glossary" marker AFTER our start marker
        end_match = end_pattern.search(full_text, match.end())
        
        if end_match:
            end_pos = end_match.start()
            raw_block = full_text[start_pos:end_pos]
            
            # Since glossary was found, we take the whole block as requested
            block = raw_block.strip()
            consumed_until = end_pos
        else:
            raw_block = full_text[start_pos:]
            
            # Fallback: If "glossary" is missing, stop at 210 words to prevent 
            # absorbing the rest of the document.
            word_count = len(raw_block.split())
            if word_count > 210:
                block, relative_end = get_word_limit_offset(raw_block, limit=210)
                consumed_until = start_pos + relative_end
            else:
                block = raw_block.strip()
                consumed_until = len(full_text)
        
        if block:
            # Skip if a number sequence (e.g. 2.4) is in the first 50 chars ---
            # We first check if the marker itself was the "\d.\d Text" variant. 
            # If it is, we bypass this rule so we don't accidentally skip valid edge cases.
            is_text_type_start = re.match(r'^\d+(?:\.\d+)+', marker_clean)
            
            skip_citation = False
            if not is_text_type_start:
                if re.search(r'\d+(?:\.\d+)+', block[:50]):
                    skip_citation = True
                    
            if not skip_citation:
                extracted_citations.append({
                    "block_name": block_label,
                    "unit_name": "Unknown/Full Document",
                    "source_pdf": source_pdf,
                    "pdf_page": "Full Text", 
                    "citations_count": 1,
                    "matched_markers": [marker_clean],
                    "citation_text": block
                })
            
        # Edge Case handler: Skip any newly matched markers that fall inside the chunk we just extracted
        next_idx = idx + 1
        while next_idx < len(start_matches) and start_matches[next_idx].start() < consumed_until:
            next_idx += 1
            
        idx = next_idx

    # Append to or create citations.json in the block directory
    if extracted_citations:
        citations_file = block_folder_path / "citations.json"
        
        if citations_file.exists():
            try:
                with open(citations_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
            existing_data.extend(extracted_citations)
        else:
            existing_data = extracted_citations
            
        with open(citations_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)
            
        print(f"✓ Extracted {len(extracted_citations)} special citation blocks to {citations_file}")
    else:
        print("No citations found matching the criteria in the provided text.")
        
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Extract alternative special-case citation blocks from raw PDF text.")
    # parser.add_argument("--raw_text_path", required=True, type=str, help="Path to the _raw_text.json file.")
    # parser.add_argument("--block_folder_path", required=True, type=str, help="Path to the output block directory (e.g., rag_store/BLOCK_NAME).")
    
    # args = parser.parse_args()
    
    # raw_text_path = Path(args.raw_text_path)
    # block_folder_path = Path(args.block_folder_path)
    
    raw_text_path = Path(r"rag_store_books\BEGC 103_ Indian Writing in English\BLOCK 3 POETRY IANOU THE PEOPLES UNIVE RSITY 27\BEGC_103_BLOCK_3_raw_text.json")
    block_folder_path = Path(r"rag_store_books\BEGC 103_ Indian Writing in English\BLOCK 3 POETRY IANOU THE PEOPLES UNIVE RSITY 27")
    
    if not raw_text_path.exists():
        print(f"Error: Raw text file not found at {raw_text_path}")
        exit(1)
        
    block_folder_path.mkdir(parents=True, exist_ok=True)
    
    process_special_citations(raw_text_path, block_folder_path)