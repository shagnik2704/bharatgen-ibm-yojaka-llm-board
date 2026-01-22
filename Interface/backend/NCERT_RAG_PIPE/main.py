import faiss
import pickle
from sentence_transformers import SentenceTransformer

# CONFIG
INDEX_PATH = "vector_db.index"
CHUNKS_PATH = "chunks_metadata.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_K = 1  # Get the single most similar chunk

class RAGRetriever:
    def __init__(self):
        # Load the model and the pre-built vector database
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)

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

def main(topic_input, theme_input):
    retriever = RAGRetriever()

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
