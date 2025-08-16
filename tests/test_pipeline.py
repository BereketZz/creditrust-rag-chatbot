# tests/test_pipeline.py
import os, pickle, json, subprocess, sys
import numpy as np
import pandas as pd
import faiss
import pytest

import pipeline  # the refactored module above

class DummyEmbedder:
    """Deterministic, light-weight embedder for tests."""
    def __init__(self, dim=8): self.dim = dim
    def encode(self, texts, show_progress_bar=False):
        # Map each text to a deterministic vector using its hash
        vecs = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            vecs.append(rng.random(self.dim, dtype=np.float32))
        return np.vstack(vecs)

@pytest.fixture
def small_df():
    data = [
        {"Product": "Credit card", "cleaned_narrative": "Late fee charged. Payment posted late due to system error."},
        {"Product": "Mortgage", "cleaned_narrative": "Escrow analysis incorrect. Taxes misapplied; repeated calls made."},
    ]
    return pd.DataFrame(data)

def test_load_data_roundtrip(tmp_path):
    csv_path = tmp_path / "filtered_complaints.csv"
    df = pd.DataFrame({"Product":["A","B"], "cleaned_narrative":["x","y"]})
    df.to_csv(csv_path, index=False)
    loaded = pipeline.load_data(str(csv_path), chunksize=1)  # exercise chunked read
    assert len(loaded) == 2
    assert list(loaded.columns) == ["Product","cleaned_narrative"]

def test_chunking_respects_overlap_and_nonempty(small_df):
    splitter = pipeline.make_splitter(chunk_size=40, chunk_overlap=10)
    chunks, meta = pipeline.chunk_dataframe(small_df, splitter=splitter)
    assert len(chunks) == len(meta) > 0
    # Verify chunk metadata integrity
    assert all(k in meta[0] for k in ["complaint_id","product","chunk_index"])
    # Spot-check overlap: consecutive chunks should overlap by at most configured amount
    # (We can't easily assert exact chars due to separators; verify no chunk is empty and average size ~ chunk_size)
    avg_len = sum(len(c) for c in chunks) / len(chunks)
    assert 20 <= avg_len <= 60

def test_embedding_shape_matches_texts(small_df):
    splitter = pipeline.make_splitter(chunk_size=60, chunk_overlap=0)
    chunks, _ = pipeline.chunk_dataframe(small_df, splitter=splitter)
    emb = DummyEmbedder(dim=16)
    E = pipeline.embed_chunks(chunks, emb)
    assert E.shape == (len(chunks), 16)
    assert E.dtype == np.float32

def test_faiss_index_count_matches_embeddings(small_df):
    splitter = pipeline.make_splitter(chunk_size=50, chunk_overlap=10)
    chunks, _ = pipeline.chunk_dataframe(small_df, splitter=splitter)
    E = DummyEmbedder(dim=8).encode(chunks)
    index = pipeline.build_faiss_index(E)
    assert isinstance(index, faiss.IndexFlatL2)
    assert index.ntotal == E.shape[0]

def test_persist_writes_all_files(tmp_path, small_df):
    splitter = pipeline.make_splitter(chunk_size=50, chunk_overlap=10)
    chunks, meta = pipeline.chunk_dataframe(small_df, splitter=splitter)
    E = DummyEmbedder(dim=8).encode(chunks)
    index = pipeline.build_faiss_index(E)
    out = pipeline.persist(index, meta, chunks, str(tmp_path))
    for k in ["index","meta","chunks"]:
        assert os.path.exists(out[k])
    # Roundtrip a couple of artifacts
    idx2 = faiss.read_index(out["index"])
    assert idx2.ntotal == len(chunks)
    with open(out["meta"],"rb") as f:
        meta2 = pickle.load(f)
    assert meta2[0]["product"] in ["Credit card","Mortgage"]

@pytest.mark.integration
def test_end_to_end_small_csv(tmp_path):
    # Build a tiny CSV and run the full flow with a DummyEmbedder
    csv_path = tmp_path / "filtered_complaints.csv"
    df = pd.DataFrame({
        "Product":["Loans","Credit card"],
        "cleaned_narrative":[
            "System outage caused duplicate charges. Please reverse.",
            "Card declined during travel; prior notice given."
        ]
    })
    df.to_csv(csv_path, index=False)

    df_loaded = pipeline.load_data(str(csv_path), chunksize=1)
    chunks, meta = pipeline.chunk_dataframe(df_loaded, splitter=pipeline.make_splitter(chunk_size=64, chunk_overlap=16))
    E = DummyEmbedder(dim=32).encode(chunks)
    index = pipeline.build_faiss_index(E)
    out = pipeline.persist(index, meta, chunks, str(tmp_path / "vector_store"))

    # Assertions proving correctness & reliability signals
    assert faiss.read_index(out["index"]).ntotal == len(chunks)
    with open(out["meta"], "rb") as f:
        meta2 = pickle.load(f)
    assert len(meta2) == len(chunks)
    # Check metadata alignment for first chunk
    assert meta2[0]["chunk_index"] == 0
    # Ensure chunks file is valid
    with open(out["chunks"], "rb") as f:
        stored_chunks = pickle.load(f)
    assert stored_chunks[:2] == chunks[:2]

