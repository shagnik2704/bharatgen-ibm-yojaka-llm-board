import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

DATA_DIR = "data"
DB_PATH = "vector_store/unified_ncert_index"

# --- 1. INITIALIZE ONCE (Saves Memory/Time) ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)

def ingest_all_books():
    all_processed_docs = []
    
    # Regex to catch: CHAPTER 1, Chapter I, CHAPTER One, Section 1, etc.
    # Added \b for boundary and better newline handling
    chapter_regex = r"(?i)\b(CHAPTER|Section)\s+([IVXLCDM\d]+|[a-zA-Z]+)"

    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: {DATA_DIR} directory not found.")
        return

    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".pdf"):
            continue
            
        book_name = os.path.splitext(filename)[0].replace("_", " ")
        print(f"📖 Processing: {book_name}...")
        
        loader = PyPDFLoader(os.path.join(DATA_DIR, filename))
        pages = loader.load()

        # Start with 'Front Matter' to avoid false 'Introduction' labels
        current_chapter = "front matter" 

        for page in pages:
            text = page.page_content
            
            # Check for chapter marker to update current_chapter
            ch_match = re.search(chapter_regex, text)
            if ch_match:
                current_chapter = f"{ch_match.group(1)} {ch_match.group(2)}".lower()
            
            # Split page text into chunks
            text_chunks = splitter.split_text(text)
            
            for chunk in text_chunks:
                # Format context clearly for the LLM
                contextual_text = f"Source: {book_name} | {current_chapter.title()}\nContent: {chunk}"
                
                all_processed_docs.append(Document(
                    page_content=contextual_text,
                    metadata={
                        "book": book_name.lower(), 
                        "chapter": current_chapter.lower(),
                        "source": filename
                    }
                ))

    # --- 2. SAVE INDEX ---
    if all_processed_docs:
        print(f"📦 Creating FAISS index for {len(all_processed_docs)} chunks...")
        vector_db = FAISS.from_documents(all_processed_docs, embeddings)
        
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        vector_db.save_local(DB_PATH)
        print(f"✅ Success! Saved index to {DB_PATH}")
    else:
        print("⚠️ No documents were processed.")

if __name__ == "__main__":
    ingest_all_books()