import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load Model with Security and Memory Fixes
MODEL_ID = "bharatgenai/Param-1-2.9B-Instruct"

# # Added trust_remote_code=True to resolve the ValueError
# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID, 
#     torch_dtype=torch.bfloat16, 
#     device_map="auto",
#     trust_remote_code=True,      # Required for Param architecture
#     low_cpu_mem_usage=True       # Helps with large model loading
# )

# 2. Load Vector DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.load_local(
    "vector_store/unified_ncert_index", 
    embeddings, 
    allow_dangerous_deserialization=True
)

def generate_rag_question(theme, topic):
    """
    Retrieves relevant NCERT chunks based on theme and topic, 
    then uses Param-1 to generate a question.
    """
    theme_clean = theme.lower().strip()
    
    # --- SMART RETRIEVAL LOGIC ---
    # Attempt 1: Filter by Book
    results = vector_db.similarity_search(topic, k=3, filter={"book": theme_clean})
    
    # Attempt 2: Filter by Chapter
    if not results:
        results = vector_db.similarity_search(topic, k=3, filter={"chapter": theme_clean})
    
    # Attempt 3: Global Semantic Fallback
    if not results:
        print(f"🔍 No metadata match for '{theme}'. Searching all books semantically...")
        results = vector_db.similarity_search(f"{theme} {topic}", k=3)

    # Combine retrieved text
    context = "\n\n".join([doc.page_content for doc in results])
    return context
    
    # --- CHAT PROMPT TEMPLATE ---
    # Param-1 uses an Instruct format. We use markers to guide it.
    prompt = f"""<|system|>
You are an expert NCERT teacher. Use the provided context to create a conceptual question.
<|user|>
CONTEXT:
{context}

TOPIC: {topic}
<|assistant|>
Question:"""
    
    # --- INFERENCE ---
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100, 
            temperature=0.4,   # Lower temperature for more factual/precise questions
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and extract only the assistant's answer
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Logic to split the output and get only the generated question
    if "Question:" in decoded_output:
        return decoded_output.split("Question:")[-1].strip()
    return decoded_output

if __name__ == "__main__":
    # Test cases to verify the logic
    print("\n--- TEST: PHYSICS ---")
    print(generate_rag_question("physics_11th", "Units and Measurements"))
    
    # print("\n--- TEST: HISTORY ---")
    # print(generate_rag_question("history_10th", "The Salt March"))