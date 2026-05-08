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
    # 1. "Chorus" or "Creon" with optional whitespace
    # 2. OR 1-2 words with AT LEAST ONE whitespace
    # Both must be preceded by a newline/start of string and followed by a colon.
    start_pattern = re.compile(
        r'(?:^|\n+)('
        r'(?:Chorus|Creon)[ \t]*'
        r'|'
        r'[a-zA-Z]+(?:[ \t]+[a-zA-Z]+)?[ \t]+'
        r'):', re.IGNORECASE
    )
    
    # regex matches: 
    # Whole words "analysis" or "summary", OR a decimal number sequence like "2.4"
    end_pattern = re.compile(r'\b(?:analysis|summary)\b|\b\d+\.\d+', re.IGNORECASE)

    # Find all potential start markers
    start_matches = list(start_pattern.finditer(full_text))
    
    extracted_citations = []
    idx = 0
    
    while idx < len(start_matches):
        match = start_matches[idx]
        speaker = match.group(1).strip()
        
        # Rule: skip if the start word is exactly 'analysis'
        if speaker.lower() == 'analysis':
            idx += 1
            continue
            
        # The start of our block includes the start marker
        start_pos = match.start()
        
        # Look for the ending "Analysis:" marker AFTER our start marker
        end_match = end_pattern.search(full_text, start_pos)
        
        if end_match:
            end_pos = end_match.start()
            raw_block = full_text[start_pos:end_pos]
        else:
            raw_block = full_text[start_pos:]
            end_pos = len(full_text)
            
        # Rule: Stop at the analysis block OR at 210 words (default)
        word_count = len(raw_block.split())
        if word_count > 210:
            block, relative_end = get_word_limit_offset(raw_block, limit=210)
            consumed_until = start_pos + relative_end
        else:
            block = raw_block.strip()
            # If we hit an end_match, the block ends there. Otherwise, it's EOF.
            consumed_until = end_pos if end_match else len(full_text)
        
        # Clean up leading/trailing whitespaces while keeping internal spacing/newlines
        block = block.strip()
        
        if block:
            # We mimic the dictionary fields output by process_pdfs.py's extract_block_citations
            extracted_citations.append({
                "block_name": block_label,
                "unit_name": "Unknown/Full Document",
                "source_pdf": source_pdf,
                "pdf_page": "Full Text",  # Replaces integer page number because this runs on full_text
                "citations_count": 1,
                "matched_markers": [f"{speaker}:"],
                "citation_text": block
            })
            
        # Edge Case handler: Skip any newly matched speakers that fall inside the chunk we just extracted
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
    # parser = argparse.ArgumentParser(description="Extract special-case citation blocks from raw PDF text.")
    # parser.add_argument("--raw_text_path", required=True, type=str, help="Path to the _raw_text.json file.")
    # parser.add_argument("--block_folder_path", required=True, type=str, help="Path to the output block directory (e.g., rag_store/BLOCK_NAME).")
    
    # args = parser.parse_args()
    
    # raw_text_path = Path(args.raw_text_path)
    # block_folder_path = Path(args.block_folder_path)
    
    raw_text_path = Path(r"rag_store_books\BEGC 102_ European Classical Literature\BLOCK 2\BEGC_102_BLOCK_2_raw_text.json")
    block_folder_path = Path(r"rag_store_books\BEGC 102_ European Classical Literature\BLOCK 2")
    
    if not raw_text_path.exists():
        print(f"Error: Raw text file not found at {raw_text_path}")
        exit(1)
        
    block_folder_path.mkdir(parents=True, exist_ok=True)
    
    process_special_citations(raw_text_path, block_folder_path)