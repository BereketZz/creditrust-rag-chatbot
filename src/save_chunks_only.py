import pandas as pd
import pickle
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configuration
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
VECTOR_STORE_DIR = "../vector_store/"
CLEANED_DATA_PATH = "data/filtered_complaints.csv"
CHUNKS_FILE = os.path.join(VECTOR_STORE_DIR, "chunks.pkl")

# Load cleaned complaints
chunks = pd.read_csv(CLEANED_DATA_PATH, chunksize=10000)
df = pd.concat(chunks, ignore_index=True)

# Initialize splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ".", " "]
)

# Chunk text
all_chunks = []

for idx, row in df.iterrows():
    original_text = row["cleaned_narrative"]
    chunks = text_splitter.split_text(original_text)
    all_chunks.extend(chunks)

# Save to pickle
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

with open(CHUNKS_FILE, "wb") as f:
    pickle.dump(all_chunks, f)

print(f"✅ all_chunks saved to {CHUNKS_FILE} ({len(all_chunks)} chunks)")
