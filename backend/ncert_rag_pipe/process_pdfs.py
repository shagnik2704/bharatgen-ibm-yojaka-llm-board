import os
import sys
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import re
from typing import List, Dict, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from .utils.pdf_loader import load_pdf
    from .utils.hierarchical_chunker import HierarchicalChunker
    from .utils.knowledge_graph_builder import build_hierarchical_kg
except ImportError:
    from utils.pdf_loader import load_pdf
    from utils.hierarchical_chunker import HierarchicalChunker
    from utils.knowledge_graph_builder import build_hierarchical_kg

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler('process_pdfs.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load the pre-trained embedding model once globally
# Switched to multilingual model to support Hindi and English embeddings
logger.info("Loading pre-trained multilingual embedding model...")
embed_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


def _safe_path_segment(text: str) -> str:
    """Normalize arbitrary extracted text into a safe folder/file name segment."""
    text = (text or "").strip()
    text = re.sub(r"[<>:\"/\\|?*]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:120] if text else "UNSPECIFIED"


def _extract_block_label(pdf_stem: str, full_text: str) -> str:
    """
    Extract block label as `BLOCK <NAME>` from filename or text.
    Evaluates both and picks the most descriptive (longest) title.
    """
    # 1. Clean filename (replace underscores and hyphens with spaces)
    stem_norm = re.sub(r"[_\-]+", " ", (pdf_stem or "")).upper()
    stem_norm = re.sub(r"\s+", " ", stem_norm).strip()
    
    # 2. Clean document text (first 6000 chars)
    text_head = re.sub(r"\s+", " ", (full_text or "")[:6000]).upper()
    
    # 3. Capture everything after "BLOCK" until a structural keyword or end of string
    regex_pattern = r"\bBLOCK\s*([\w\s:,-]{1,100}?)(?=\b(?:UNIT|CHAPTER|STRUCTURE|OBJECTIVES|INTRODUCTION|P\d)\b|$)"
    
    m_file = re.search(regex_pattern, stem_norm)
    m_text = re.search(regex_pattern, text_head)
    
    label_file = m_file.group(1).strip() if m_file else ""
    label_text = m_text.group(1).strip() if m_text else ""
    
    # 4. Pick the richer/longer label 
    # (e.g., "1 KALIDASA" from text beats "1" from filename)
    best_label = label_text if len(label_text) > len(label_file) else label_file
    
    if best_label:
        return f"BLOCK {_safe_path_segment(best_label.upper())}"
        
    return "BLOCK UNSPECIFIED"


def _build_base_unit_document_id(pdf_stem: str, doc_hash: str) -> str:
    """
    Build a unit-compatible base id while avoiding `...BLOCK..._UNIT_...` repetition.
    Keep `Unit-` prefix for compatibility with existing retriever conventions.
    """
    base = re.sub(r"(?i)\bblock\s*[A-Za-z0-9-]+", "", pdf_stem or "")
    base = re.sub(r"\s+", " ", base).strip().replace(" ", "_")
    base = _safe_path_segment(base)
    return f"Unit-{base}_{doc_hash}"


def _rebuild_children(chunks):
    """Recompute children_ids after any filtering/merge operation."""
    by_id = {c.chunk_id: c for c in chunks}
    for c in chunks:
        c.children_ids = []
    for c in chunks:
        if c.parent_id and c.parent_id in by_id:
            by_id[c.parent_id].children_ids.append(c.chunk_id)


def _repair_missing_parents(chunks):
    """Rewire dangling parent links to nearest available ancestor or root."""
    if not chunks:
        return

    by_id = {c.chunk_id: c for c in chunks}
    root_id = chunks[0].chunk_id

    for c in chunks:
        if c.parent_id is None:
            continue
        if c.parent_id in by_id:
            continue

        # Fallback to nearest level-1 unit chunk, otherwise root.
        candidate_parent = None
        if c.level > 1:
            for prev in reversed(chunks):
                if prev.chunk_id == c.chunk_id:
                    continue
                if prev.level < c.level and prev.chunk_id in by_id:
                    candidate_parent = prev.chunk_id
                    break

        c.parent_id = candidate_parent or root_id


def _post_process_chunks(chunks, min_words: int = 12):
    """Merge/drop low-value chunks after primary chunking."""
    if not chunks:
        return chunks

    low_value_title_re = re.compile(r"(?i)^(?:document root / introduction|structure|contents?)$")
    numeric_only_re = re.compile(r"^\s*[\d\s.]+\s*$")

    kept = []
    seen_title_text = set()

    for chunk in chunks:
        text = (chunk.text or "").strip()
        title = (chunk.title or "").strip()
        wc = len(text.split())

        dedup_key = (re.sub(r"\s+", " ", title.lower()), re.sub(r"\s+", " ", text.lower()[:400]))
        if dedup_key in seen_title_text:
            continue
        seen_title_text.add(dedup_key)

        # FIX: Removed `or numeric_only_re.match(title) is not None`
        # A numeric title is fine as long as the chunk has actual text > min_words
        is_low_value = (
            wc < min_words
            or low_value_title_re.match(title) is not None
            or numeric_only_re.match(text) is not None
        )

        if is_low_value and kept:
            prev = kept[-1]
            if prev.parent_id == chunk.parent_id or prev.level == chunk.level:
                prev.text = (prev.text + "\n\n" + text).strip() if text else prev.text
            continue

        kept.append(chunk)

    _repair_missing_parents(kept)
    _rebuild_children(kept)
    return kept


def _index_single_document(
    document_payload: dict,
    pdf_path: Path,
    output_dir: Path,
    pages: List[str],
    force_ocr: bool = False,
    block_label: Optional[str] = None,
) -> Optional[dict]:
    """Index one document payload (full PDF or pre-split unit) into the RAG store."""
    document_id = document_payload["document_id"]
    document_text = document_payload["text"]

    chunker = HierarchicalChunker()
    chunks = chunker.chunk_document(pages, document_text, document_id)
    chunks = _post_process_chunks(chunks)

    if not chunks:
        logger.error(f"No chunks generated for {document_id}")
        return None

    logger.info(f"Building knowledge graph for {document_id}...")
    graph = build_hierarchical_kg(chunks)

    logger.info(f"Computing embeddings for {document_id}...")
    texts = [f"{c.title}: {c.text[:1000]}" for c in chunks]
    embeddings = embed_model.encode(texts, normalize_embeddings=True).astype(np.float32)

    doc_dir = output_dir / document_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    with open(doc_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in chunks], f, indent=2, ensure_ascii=False)

    graph_path = doc_dir / "graph.pkl"
    with open(graph_path, "wb") as f:
        pickle.dump(graph, f)

    try:
        try:
            from .utils.visualize_kg import visualize_hierarchy
        except ImportError:
            from utils.visualize_kg import visualize_hierarchy
        image_path = doc_dir / "graph_visualization.png"
        visualize_hierarchy(str(graph_path), str(image_path))
    except Exception as vis_err:
        logger.warning(f"Could not generate graph image for {document_id}: {vis_err}")

    np.save(doc_dir / "embeddings.npy", embeddings)

    with open(doc_dir / "id_index.json", "w") as f:
        json.dump({str(i): c.chunk_id for i, c in enumerate(chunks)}, f)

    metadata = {
        "document_id": document_id,
        "unit_title": document_payload.get("title", ""),
        "block_label": block_label,
        "source_pdf": pdf_path.name,
        "processed_at": datetime.now().isoformat(),
        "total_pages": len(pages),
        "total_chunks": len(chunks),
        "total_words": sum(len(c.text.split()) for c in chunks),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "store_path": str(doc_dir),
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "embedding_dim": embeddings.shape[1],
        "force_ocr": force_ocr,
    }

    with open(doc_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✓ Saved {len(chunks)} cleaned chunks to {doc_dir}/")
    return metadata


def process_single_pdf(
    pdf_path: Path,
    output_dir: Path,
    force_ocr: bool = False,
    split_by_unit: bool = True,
) -> List[dict]:
    """Process a single PDF and save it in the designated output directory."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Processing: {pdf_path.name}")
    logger.info(f"{'='*70}")
    
    try:
        pages, full_text = load_pdf(str(pdf_path), force_ocr=force_ocr)
        
        doc_hash = hashlib.md5(str(pdf_path).encode()).hexdigest()[:8]
        # Keep Unit-* for compatibility, but avoid embedding BLOCK in unit id repeatedly.
        base_document_id = _build_base_unit_document_id(pdf_path.stem, doc_hash)

        # Create explicit block folder level: subject/egyankosh/BLOCK <NAME>/Unit-.../chunks.
        block_label = _extract_block_label(pdf_path.stem, full_text)
        block_output_dir = output_dir / block_label
        block_output_dir.mkdir(parents=True, exist_ok=True)

        chunker = HierarchicalChunker()
        if split_by_unit:
            logger.info("Applying upstream unit-level splitting before indexing...")
            document_payloads = chunker.split_into_unit_documents(full_text, base_document_id)
        else:
            document_payloads = [{"document_id": base_document_id, "title": "Full Document", "text": full_text}]

        metadata_list = []
        for payload in document_payloads:
            metadata = _index_single_document(
                payload,
                pdf_path,
                block_output_dir,
                pages,
                force_ocr=force_ocr,
                block_label=block_label,
            )
            if metadata:
                metadata_list.append(metadata)

        return metadata_list
        
    except Exception as e:
        logger.error(f"Error processing {pdf_path.name}: {e}", exc_info=True)
        return []


def batch_process(input_dir: str, output_dir: str = None, force_ocr: bool = False, split_by_unit: bool = True):
    """Batch process all PDFs, preserving nested folder structures."""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logger.error(f"Input path not found: {input_dir}")
        sys.exit(1)
        
        
    # Auto-name output directory if none is provided
    if not output_dir:
        output_path = Path(f"rag_store_{input_path.name}" if input_path.is_dir() else "rag_store")
    else:
        output_path = Path(output_dir)
        
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Handle both single file and recursive directory scanning
    if input_path.is_file():
        pdfs = [input_path]
        base_input_path = input_path.parent
    else:
        pdfs = sorted(input_path.rglob("*.pdf"))  
        base_input_path = input_path
    
    logger.info(f"\n{'='*70}")
    logger.info(f"  Hierarchical Multi-PDF RAG Orchestrator (100% Local)")
    logger.info(f"{'='*70}")
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"PDFs:   {len(pdfs)}\n")
    
    all_metadata = {}
    
    for i, pdf_path in enumerate(pdfs, 1):
        logger.info(f"\n[{i}/{len(pdfs)}] Processing: {pdf_path.relative_to(base_input_path)}")
        
        rel_path = pdf_path.relative_to(base_input_path)
        
        # Strip 'egyankosh' from the nested folder path
        clean_parent_parts = [p for p in rel_path.parent.parts if p.lower() != "egyankosh"]
        target_dir = output_path.joinpath(*clean_parent_parts)
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Process without API client
        metadata_list = process_single_pdf(pdf_path, target_dir, force_ocr=force_ocr, split_by_unit=split_by_unit)
        for metadata in metadata_list:
            all_metadata[metadata["document_id"]] = metadata
    
    master = {
        "timestamp": datetime.now().isoformat(),
        "input_source": str(input_path),
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "total_documents": len(all_metadata),
        "documents": all_metadata,
    }
    
    with open(output_path / "master_index.json", "w") as f:
        json.dump(master, f, indent=2)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"✓ Successfully processed {len(all_metadata)} PDFs")
    logger.info(f"✓ Master index: {output_path}/master_index.json")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default='books', required=False, help="Input PDF or nested directory of PDFs")
    parser.add_argument("--output", "-o", default=None, help="Output directory (defaults to rag_store_<input_name>)")
    parser.add_argument("--force_ocr", action="store_true", help="Force OCR extraction on all pages")
    parser.add_argument("--no_split_by_unit", action="store_true", help="Disable unit-level pre-splitting before indexing")
    args = parser.parse_args()
    
    batch_process(args.input, args.output, force_ocr=args.force_ocr, split_by_unit=not args.no_split_by_unit)
