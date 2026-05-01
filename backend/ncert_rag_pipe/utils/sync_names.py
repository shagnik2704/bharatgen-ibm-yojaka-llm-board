"""
Some block folder names of the indexed data need to be tweaked manually
because a few names like 'BLOCK CILAPPATIKARAN' do not get inferred automatically.
Thus, the purpose of this script is to synchronize paths and names of the indexed data.
It keeps block folder names, 'citations.json', 'metadata.json', and 'master_index.json' consistent.
Run this script after you manually change one or more block folder names.
"""

import os
import json
from pathlib import Path
from datetime import datetime

def main(root_directory):
    base_path = Path(root_directory).resolve()
    master_index_path = base_path / "master_index.json"
    
    if not base_path.exists():
        print(f"❌ Error: The directory {base_path} does not exist.")
        return

    # 1. Initialize a fresh master dictionary to guarantee it perfectly reflects the file system
    master_data = {
        "timestamp": datetime.now().isoformat(),
        "input_source": str(base_path),
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "total_documents": 0,
        "documents": {}
    }
    
    print(f"\nScanning directory: {base_path}\n" + "="*60)
    
    # 2. Iterate through Subject Folders (Depth 1)
    for subject_dir in base_path.iterdir():
        if not subject_dir.is_dir() or subject_dir.name.startswith('.'): 
            continue
        
        # 3. Iterate through Block Folders (Depth 2)
        for block_dir in subject_dir.iterdir():
            if not block_dir.is_dir() or block_dir.name.startswith('.'): 
                continue
            
            # The exact name of the folder is our single source of truth
            new_block_label = block_dir.name
            print(f"\n📁 Processing folder: '{new_block_label}' (in {subject_dir.name})")

            # --- Update citations.json ---
            cit_file = block_dir / "citations.json"
            if cit_file.exists():
                try:
                    with open(cit_file, 'r', encoding='utf-8') as f:
                        cit_data = json.load(f)
                    
                    updated_cit = False
                    if isinstance(cit_data, list):
                        for item in cit_data:
                            if item.get("block_name") != new_block_label:
                                item["block_name"] = new_block_label
                                updated_cit = True
                    
                    if updated_cit:
                        with open(cit_file, 'w', encoding='utf-8') as f:
                            json.dump(cit_data, f, indent=2, ensure_ascii=False)
                        print(f"  [SUCCESS] Synced citations.json -> '{new_block_label}'")
                except Exception as e:
                    print(f"  [ERROR] Failed processing {cit_file.name}: {e}")

            # --- Update metadata.json AND Reconstruct Master Index ---
            for meta_file in block_dir.rglob("metadata.json"):
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta_data = json.load(f)
                    
                    # Grab the actual, robust path from the file system
                    current_store_path = str(meta_file.parent)
                    
                    needs_update = False
                    
                    if meta_data.get("block_label") != new_block_label:
                        meta_data["block_label"] = new_block_label
                        needs_update = True
                        
                    if meta_data.get("store_path") != current_store_path:
                        meta_data["store_path"] = current_store_path
                        needs_update = True
                        
                    if needs_update:
                        # Write local fix
                        with open(meta_file, 'w', encoding='utf-8') as f:
                            json.dump(meta_data, f, indent=2, ensure_ascii=False)
                        print(f"  [SUCCESS] Synced {meta_file.parent.name}/metadata.json (Label & Path)")
                    
                    # Add this document to our reconstructed Master Index
                    doc_id = meta_data.get("document_id")
                    if doc_id:
                        master_data["documents"][doc_id] = meta_data
                            
                except Exception as e:
                    print(f"  [ERROR] Failed processing {meta_file.name}: {e}")

    # 4. Finalize and Save the Master Index back to disk
    master_data["total_documents"] = len(master_data["documents"])
    print("\n" + "="*60)
    
    try:
        with open(master_index_path, 'w', encoding='utf-8') as f:
            json.dump(master_data, f, indent=2, ensure_ascii=False)
        print(f"✅ [SUCCESS] Reconstructed master_index.json with {master_data['total_documents']} documents.")
    except Exception as e:
        print(f"❌ [ERROR] Failed to save master_index.json: {e}")

if __name__ == "__main__":
    # Ensure this points to the root of your rag_store output
    ROOT_DIR = "rag_store_books" 
    main(ROOT_DIR)