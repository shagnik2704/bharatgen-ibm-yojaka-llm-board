import faiss
import pickle
from sentence_transformers import SentenceTransformer

# CONFIG
INDEX_PATH = "vector_db.index"
CHUNKS_PATH = "chunks_metadata.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_K = 1  # Set your desired default here

class RAGRetriever:
    def __init__(self):
        # Load the model and the pre-built vector database
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)

    def get_context(self, query, k=DEFAULT_K):
        """
        Retrieves top K chunks and merges them into one string.
        Returns the merged text and the best similarity score.
        """
        query_vec = self.model.encode([query])
        distances, indices = self.index.search(query_vec, k)
        
        retrieved_texts = []
        for i in range(k):
            idx = indices[0][i]
            if idx != -1: # Ensure valid index
                retrieved_texts.append(self.chunks[idx])
        
        # Calculate similarity score for the #1 result
        best_score = 1 / (1 + distances[0][0])
        
        return "\n\n".join(retrieved_texts), best_score

def main():
    retriever = RAGRetriever()

    # 1. Inputs
    topic_input = input("Enter Topic: ").strip()
    theme_input = input("Enter Theme: ").strip()

    # 2. Retrieval using default K
    topic_chunk, t_score = retriever.get_context(topic_input)
    theme_chunk, th_score = retriever.get_context(theme_input)

    # 3. Display Results
    print("\n" + "="*60)
    print(f"📊 RETRIEVAL REPORT (K={DEFAULT_K})")
    print("-" * 60)
    print(f"🔵 TOPIC: {topic_input} (Match: {t_score:.2%})")
    print(f"🟢 THEME: {theme_input} (Match: {th_score:.2%})")
    print("="*60 + "\n")

    # This dictionary is what you would pass to your LLM Prompt
    return {
        "topic_text": topic_chunk,
        "theme_text": theme_chunk
    }

if __name__ == "__main__":
    context_data = main()
    
    # You can now access:
    # context_data['topic_text'] 
    # context_data['theme_text']