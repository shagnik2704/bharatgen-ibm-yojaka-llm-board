import faiss
import pickle
from sentence_transformers import SentenceTransformer

# CONFIG
INDEX_PATH = "vector_db.index"
CHUNKS_PATH = "chunks_metadata.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

class RAGRetriever:
    def __init__(self):
        # Load the model, index, and text metadata
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)

    def get_specific_chunk(self, query):
        query_vec = self.model.encode([query])
        # Search for TOP_K = 1
        distances, indices = self.index.search(query_vec, 1)
        
        idx = indices[0][0]
        return self.chunks[idx]

def main():
    retriever = RAGRetriever()

    topic = input("Enter Topic: ").strip()
    theme = input("Enter Theme: ").strip()

    # Get chunks separately
    topic_chunk = retriever.get_specific_chunk(topic)
    theme_chunk = retriever.get_specific_chunk(theme)

    print("\n" + "="*50)
    print(f"🔵 TOPIC CHUNK FOR: {topic}")
    print(topic_chunk[:500] + "...") # Preview first 500 chars
    
    print("\n" + "="*50)
    print(f"🟢 THEME CHUNK FOR: {theme}")
    print(theme_chunk[:500] + "...") # Preview first 500 chars

    return topic_chunk, theme_chunk

if __name__ == "__main__":
    t_chunk, th_chunk = main()
    # You can now pass these into your Prompt template