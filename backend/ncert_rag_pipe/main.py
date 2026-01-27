import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))  # Go up two levels: ncert_rag_pipe -> backend -> root

# Build absolute paths to the database files (in indexes folder at project root)
INDEXES_DIR = os.path.join(PROJECT_ROOT, "indexes")
INDEX_PATH = os.path.join(INDEXES_DIR, "vector_db.index")
CHUNKS_PATH = os.path.join(INDEXES_DIR, "chunks_metadata.pkl")

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_K = 1

# Global RAG retriever instance (cached to avoid reloading on every call)
_rag_retriever = None

class RAGRetriever:
    def __init__(self):
        # Load the model and the pre-built vector database
        print("Loading RAG retriever (this happens once)...")
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)
        print("RAG retriever loaded successfully!")

    def get_context(self, query, k=DEFAULT_K):
        """
        Retrieves the top K most similar chunks.
        Returns the combined text and the best similarity score.
        """
        query_vec = self.model.encode([query])
        distances, indices = self.index.search(query_vec, k)
        
        retrieved_texts = []
        for i in range(k):
            idx = indices[0][i]
            if idx != -1:
                retrieved_texts.append(self.chunks[idx])
        
        # Calculate similarity score for the #1 result (Distance to Similarity)
        best_score = 1 / (1 + distances[0][0])
        
        return "\n\n".join(retrieved_texts), best_score

def get_retriever():
    """Get or create the global RAG retriever instance (singleton pattern)."""
    global _rag_retriever
    if _rag_retriever is None:
        _rag_retriever = RAGRetriever()
    return _rag_retriever

def main(topic_input, theme_input):
    retriever = get_retriever()

    # 2. Retrieval
    topic_chunk, t_score = retriever.get_context(topic_input)
    theme_chunk, th_score = retriever.get_context(theme_input)

    # 3. Output Full Results
    print("\n" + "="*80)
    print(f"📄 FULL RETRIEVAL DATA")
    print("="*80)

    print(f"\n🔵 TOPIC CHUNK (Similarity: {t_score:.2%})")
    print("-" * 40)
    print(topic_chunk) # This prints the full ~1000 word chunk

    print(f"\n🟢 THEME CHUNK (Similarity: {th_score:.2%})")
    print("-" * 40)
    print(theme_chunk) # This prints the full ~1000 word chunk
    
    print("\n" + "="*80)

    # Return as a dictionary for your Assessment Prompt logic
    return topic_chunk, theme_chunk
