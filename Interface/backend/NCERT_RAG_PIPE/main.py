import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load Model (Using Param-1-2.9B as requested)
MODEL_ID = "bharatgenai/Param-1-2.9B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")

# Load Vector DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.load_local("vector_store/unified_ncert_index", embeddings, allow_dangerous_deserialization=True)

def generate_rag_question(theme, topic):
    """
    Finds chunks based on theme/topic and generates a question.
    'Theme' is used for filtering; if no exact match is found, 
    it falls back to a semantic search across everything.
    """
    theme_clean = theme.lower()
    
    # Try filtering by metadata first (Book or Chapter)
    results = []
    # Check if theme matches a book
    results = vector_db.similarity_search(topic, k=3, filter={"book": theme_clean})
    
    # If no book found, check if it matches a chapter
    if not results:
        results = vector_db.similarity_search(topic, k=3, filter={"chapter": theme_clean})
    
    # FALLBACK: If still no results, perform a global semantic search including the theme
    if not results:
        print(f"🔍 No exact metadata match for '{theme}'. Performing global semantic search...")
        results = vector_db.similarity_search(f"{theme} {topic}", k=3)

    context = "\n\n".join([doc.page_content for doc in results])
    
    # Generate Prompt
    prompt = f"System: Use the NCERT context below to generate a conceptual question.\nContext: {context}\nTopic: {topic}\nQuestion:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=120, temperature=0.7)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("Question:")[-1].strip()

# # Example Test
# if __name__ == "__main__":
#     # Test 1: By Book
#     print("Test 1 (Book):", generate_rag_question("physics 11th", "Newton's laws"))
    
#     # Test 2: By Chapter
#     print("Test 2 (Chapter):", generate_rag_question("chapter i", "Rise of Nationalism"))