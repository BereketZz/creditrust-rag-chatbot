import pandas as pd
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# ------------------------
# Configuration
# ------------------------

# Chunking
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# File paths
CLEANED_DATA_PATH = "data/filtered_complaints.csv"
VECTOR_STORE_DIR = "../vector_store/"
INDEX_FILE = os.path.join(VECTOR_STORE_DIR, "faiss_index.bin")
META_FILE = os.path.join(VECTOR_STORE_DIR, "metadata.pkl")

# Create vector store dir if missing
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# ------------------------
# Step 1: Load Cleaned Data
# ------------------------

# Efficient, safe CSV reading
try:
    chunks = pd.read_csv("data/filtered_complaints.csv", chunksize=10000)
    df = pd.concat(chunks, ignore_index=True)
    print(f"Loaded {len(df)} complaints.")
except Exception as e:
    print("Error loading CSV:", e)
    exit()

# ------------------------
# Step 2: Chunk the Text
# ------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ".", " "]
)

# Store all chunks and metadata
all_chunks = []
metadata_list = []

print("Splitting complaints into chunks...")
for idx, row in df.iterrows():
    complaint_id = idx
    product = row["Product"]
    original_text = row["cleaned_narrative"]

    # Chunk this complaint
    chunks = text_splitter.split_text(original_text)

    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        metadata_list.append({
            "complaint_id": complaint_id,
            "product": product,
            "chunk_index": i
        })

print(f"Total text chunks created: {len(all_chunks)}")

# ------------------------
# Step 3: Embedding
# ------------------------

print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

print("Generating embeddings...")
embeddings = embedder.encode(all_chunks, show_progress_bar=True)

# ------------------------
# Step 4: FAISS Indexing
# ------------------------

embedding_dim = embeddings[0].shape[0]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)

print(f"FAISS index created with {index.ntotal} vectors.")

# ------------------------
# Step 5: Persist Index and Metadata
# ------------------------

faiss.write_index(index, INDEX_FILE)

with open(META_FILE, "wb") as f:
    pickle.dump(metadata_list, f)

print(f"FAISS index saved to {INDEX_FILE}")
print(f"Metadata saved to {META_FILE}")