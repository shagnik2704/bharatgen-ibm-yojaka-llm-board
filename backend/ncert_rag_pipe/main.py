import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))  # Go up two levels: ncert_rag_pipe -> backend -> root

# Build absolute paths to the database files (in indexes folder at project root)
INDEXES_DIR = os.path.join(PROJECT_ROOT, "indexes")

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_K = 1

def _index_paths(language: str) -> tuple[str, str]:
    """
    Language-aware index paths.
    Expects:
      indexes/<language>/vector_db.index
      indexes/<language>/chunks_metadata.pkl
    """
    lang_dir = os.path.join(INDEXES_DIR, language)
    return (
        os.path.join(lang_dir, "vector_db.index"),
        os.path.join(lang_dir, "chunks_metadata.pkl"),
    )


def _resolve_index_paths(language: str) -> tuple[str, str]:
    """
    Resolve index paths with fallback for legacy flat layout.
    - Prefer indexes/<lang>/vector_db.index when it exists.
    - For "en" only: fall back to flat indexes/vector_db.index + chunks_metadata.pkl
      when indexes/en/ does not exist.
    - Otherwise raise FileNotFoundError with a clear message.
    """
    index_path, chunks_path = _index_paths(language)
    if os.path.isfile(index_path) and os.path.isfile(chunks_path):
        return index_path, chunks_path
    # Legacy flat layout (indexes/vector_db.index, indexes/chunks_metadata.pkl)
    flat_index = os.path.join(INDEXES_DIR, "vector_db.index")
    flat_chunks = os.path.join(INDEXES_DIR, "chunks_metadata.pkl")
    if language == "en" and os.path.isfile(flat_index) and os.path.isfile(flat_chunks):
        return flat_index, flat_chunks
    if language == "hi":
        raise FileNotFoundError(
            "Hindi RAG index not found at indexes/hi/vector_db.index. "
            "Add NCERT books under data/Hindi or data/hi, then run: python backend/ncert_rag_pipe/ingest.py"
        )
    raise FileNotFoundError(
        f"RAG index for language '{language}' not found. "
        f"Expected indexes/{language}/vector_db.index (or, for 'en' only, flat indexes/vector_db.index). "
        "Run ingestion: python backend/ncert_rag_pipe/ingest.py (see README)."
    )

# Global RAG retrievers (cached per language)
_rag_retrievers: dict[str, "RAGRetriever"] = {}

class RAGRetriever:
    def __init__(self, language: str):
        self.language = language
        index_path, chunks_path = _resolve_index_paths(language)
        print(f"Loading RAG retriever for language='{language}' (this happens once)...")
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        print(f"RAG retriever loaded successfully for language='{language}'!")

    def _chunk_text(self, chunk):
        """Extract text from chunk; chunk may be dict or legacy str."""
        if isinstance(chunk, dict):
            return chunk.get("text", "")
        return chunk if isinstance(chunk, str) else ""

    def _chunk_meta(self, chunk):
        """Extract metadata from chunk; chunk may be dict or legacy str."""
        if isinstance(chunk, dict):
            return {"source_path": chunk.get("source_path"), "page": chunk.get("page")}
        return {}

    def get_context(self, query, k=DEFAULT_K):
        """
        Retrieves the top K most similar chunks.
        Returns (combined_text, best_score, metadata_list).
        metadata_list[i] = {source_path, page} for retrieved chunk i (or {} for legacy string chunks).
        """
        query_vec = self.model.encode([query])
        distances, indices = self.index.search(query_vec, k)
        
        retrieved_texts = []
        metadata_list = []
        for i in range(k):
            idx = indices[0][i]
            if idx != -1:
                raw = self.chunks[idx]
                retrieved_texts.append(self._chunk_text(raw))
                metadata_list.append(self._chunk_meta(raw))
        
        best_score = 1 / (1 + distances[0][0])
        return "\n\n".join(retrieved_texts), best_score, metadata_list

def get_retriever(language: str = "en"):
    """Get or create the global RAG retriever instance for a given language."""
    global _rag_retrievers
    if language not in _rag_retrievers:
        _rag_retrievers[language] = RAGRetriever(language=language)
    return _rag_retrievers[language]

def main(topic_input, theme_input, language: str = "en"):
    retriever = get_retriever(language=language)

    # 2. Retrieval (get_context returns text, score, metadata_list)
    topic_chunk, t_score, topic_meta = retriever.get_context(topic_input)
    theme_chunk, th_score, theme_meta = retriever.get_context(theme_input)

    # 3. Output Full Results
    print("\n" + "="*80)
    print(f"📄 FULL RETRIEVAL DATA")
    print("="*80)

    print(f"\n🔵 TOPIC CHUNK (Similarity: {t_score:.2%})")
    print("-" * 40)
    print(topic_chunk)

    print(f"\n🟢 THEME CHUNK (Similarity: {th_score:.2%})")
    print("-" * 40)
    print(theme_chunk)
    
    print("\n" + "="*80)

    return topic_chunk, theme_chunk, topic_meta, theme_meta
