import os
import faiss
import pickle
import glob
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "data"
CHUNK_SIZE = 1000 
MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "vector_db.index"
CHUNKS_PATH = "chunks_metadata.pkl"

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def read_pdf(path):
    """Extracts text from a single PDF file."""
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + " "
        return text
    except Exception as e:
        print(f"⚠️ Error reading {path}: {e}")
        return ""

def chunk_text(text, chunk_size=1000):
    """Splits text into chunks of specified word count."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# -----------------------------
# MAIN INGESTION LOGIC
# -----------------------------
def run_ingestion():
    print("🚀 Starting Automated Ingestion...")
    
    # 1. Find all PDFs in the data folder
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in '{DATA_DIR}' folder.")
        return

    print(f"📂 Found {len(pdf_files)} files: {[os.path.basename(f) for f in pdf_files]}")

    # 2. Extract and Chunk
    all_chunks = []
    for path in pdf_files:
        print(f"📖 Processing: {os.path.basename(path)}...")
        file_text = read_pdf(path)
        file_chunks = chunk_text(file_text, CHUNK_SIZE)
        all_chunks.extend(file_chunks)
    
    print(f"📦 Total chunks created: {len(all_chunks)}")

    # 3. Generate Embeddings
    print("🧠 Generating embeddings (this may take a moment)...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    
    # 4. Create and Save FAISS Index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    
    # 5. Save Text Metadata
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)
    
    print(f"✅ Success! Index saved to {INDEX_PATH}")

if __name__ == "__main__":
    # Ensure data directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 Created '{DATA_DIR}' folder. Place your PDFs there and re-run.")
    else:
        run_ingestion()