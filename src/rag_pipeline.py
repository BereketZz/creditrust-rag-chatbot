import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# -------------------------------
# Configuration
# -------------------------------

VECTOR_STORE_DIR = "vector_store"
INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index.bin")
META_PATH = os.path.join(VECTOR_STORE_DIR, "metadata.pkl")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

# -------------------------------
# Load Embeddings & Metadata
# -------------------------------

print("Loading FAISS index and metadata...")
index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "rb") as f:
    metadata_list = pickle.load(f)

CHUNKS_PATH = os.path.join(VECTOR_STORE_DIR, "chunks.pkl")

with open(CHUNKS_PATH, "rb") as f:
    all_chunks = pickle.load(f)


embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# -------------------------------
# Prompt Template
# -------------------------------

PROMPT_TEMPLATE = """
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer.

If the context doesn't contain the answer, clearly say: "I don’t have enough information."

Context:
{context}

Question: {question}
Answer:
"""
from transformers import pipeline

# Load a text generation model (small, fast, and works offline)
generator = pipeline("text-generation", model="gpt2")


# -------------------------------
# Core RAG Function
# -------------------------------

def generate_answer(question: str) -> str:
    # Step 1: Embed the question
    question_embedding = embedder.encode([question])
    
    # Step 2: Perform vector search
    distances, indices = index.search(np.array(question_embedding).astype("float32"), TOP_K)
    
    # Step 3: Retrieve top chunks and metadata
    retrieved_chunks = [all_chunks[i] for i in indices[0]]
    retrieved_metadata = [metadata_list[i] for i in indices[0]]
    
    # Step 4: Build context string
    context = "\n\n".join(retrieved_chunks)
    
    # Step 5: Construct the prompt
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    # Step 6: Generate the answer from the LLM
    response = generator(prompt, do_sample=False)[0]['generated_text']
    
    # Step 7: Trim extra prompt if needed
    return response.strip().split("Answer:")[-1].strip(), retrieved_chunks

# -------------------------------
# For Testing or CLI Use
# -------------------------------

if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        answer, sources = generate_answer(query)
        print("\n🔹 Answer:\n", answer)
        print("\n📚 Top Source Chunks:\n")
        for i, chunk in enumerate(sources):
            print(f"--- Chunk {i+1} ---\n{chunk[:300]}...\n")