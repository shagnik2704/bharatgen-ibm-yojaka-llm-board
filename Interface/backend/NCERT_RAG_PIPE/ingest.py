import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

DATA_DIR = "data"
DB_PATH = "vector_store/unified_ncert_index"

def ingest_all_books():
    all_processed_docs = []
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 1. Loop through every PDF in the data folder
    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".pdf"):
            continue
            
        # Extract Book Name from filename (e.g., Physics_11th)
        book_name = os.path.splitext(filename)[0].replace("_", " ")
        print(f"📖 Processing Book: {book_name}...")
        
        loader = PyPDFLoader(os.path.join(DATA_DIR, filename))
        pages = loader.load()

        current_chapter = "Introduction"
        
        # Regex to catch: CHAPTER 1, Chapter I, CHAPTER One, etc.
        chapter_regex = r"(?i)(CHAPTER|Section)\s*[\n\r]*\s*([IVXLCDM\d]+|[a-zA-Z]+)"

        for page in pages:
            text = page.page_content
            
            # Update current chapter if a marker is found
            ch_match = re.search(chapter_regex, text)
            if ch_match:
                current_chapter = f"{ch_match.group(1)} {ch_match.group(2)}".lower()
            
            # Split page into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
            text_chunks = splitter.split_text(text)
            
            for chunk in text_chunks:
                # INJECT Context: This makes the "Theme" searchable inside the text
                contextual_text = f"[Book: {book_name}] [Chapter: {current_chapter}] Content: {chunk}"
                
                all_processed_docs.append(Document(
                    page_content=contextual_text,
                    metadata={"book": book_name.lower(), "chapter": current_chapter.lower()}
                ))

    # 2. Save the massive unified index
    if all_processed_docs:
        vector_db = FAISS.from_documents(all_processed_docs, embeddings)
        vector_db.save_local(DB_PATH)
        print(f"✅ Success! Indexed {len(all_processed_docs)} chunks from all books.")

if __name__ == "__main__":
    ingest_all_books()