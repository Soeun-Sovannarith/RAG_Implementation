import os
import glob
import json
import math
import pandas as pd
from typing import List, Dict, Any
from pypdf import PdfReader
from splitter import split_documents
from ollama_client import get_embedding

DB_FILE_PATH = "local_vector_db.json"

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Computes mathematical Cosine Similarity between two vectors:
    similarity = dot(A, B) / (||A|| * ||B||)
    """
    if not vec_a or not vec_b:
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)

class PureVectorStore:
    """
    A lightweight, pure Python Vector Database with JSON persistence and Cosine Similarity.
    """
    def __init__(self, db_path: str = DB_FILE_PATH):
        self.db_path = db_path
        self.documents: List[Dict[str, Any]] = [] # [{"content": str, "metadata": dict, "embedding": list[float]}]
        self.load()

    def add_documents(self, docs: List[Dict[str, Any]]):
        """Generates embeddings for documents and appends them to the store."""
        for doc in docs:
            text = doc.get("content", "").strip()
            if not text:
                continue
            emb = get_embedding(text)
            if emb:
                self.documents.append({
                    "content": text,
                    "metadata": doc.get("metadata", {}),
                    "embedding": emb
                })
        self.save()

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Embeds query, computes cosine similarity against all items, and returns top-k nearest docs.
        """
        if not self.documents:
            return []
            
        query_emb = get_embedding(query)
        if not query_emb:
            return []
            
        scored_docs = []
        for doc in self.documents:
            score = cosine_similarity(query_emb, doc["embedding"])
            scored_docs.append({
                "content": doc["content"],
                "metadata": doc["metadata"],
                "score": score
            })
            
        # Sort by similarity score descending
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:k]

    def save(self):
        """Persists database state to a JSON file."""
        try:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving vector store to {self.db_path}: {e}")

    def load(self):
        """Loads database from disk if it exists."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
            except Exception as e:
                print(f"Error loading vector store from {self.db_path}: {e}")
                self.documents = []
        else:
            self.documents = []

    def clear(self):
        """Clears all records in memory and on disk."""
        self.documents = []
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

def parse_file_to_documents(path: str) -> List[Dict[str, Any]]:
    """Parses files (CSV, PDF, TXT, MD) into standard dictionary documents."""
    docs = []
    name = os.path.basename(path)
    if os.path.isdir(path):
        return docs
        
    # 1. CSV Files
    if path.endswith(".csv"):
        try:
            df = pd.read_csv(path)
            for _, r in df.iterrows():
                text = f"{r.get('Title', '')} - {r.get('Review', r.get('text', str(r.to_dict())))}"
                docs.append({"content": text.strip("- "), "metadata": {"source": name}})
        except Exception as e:
            print(f"Warning: Failed to parse CSV {name}: {e}")
            
    # 2. PDF Files
    elif path.endswith(".pdf"):
        try:
            reader = PdfReader(path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append({
                        "content": text.strip(),
                        "metadata": {"source": name, "page": page_num + 1}
                    })
        except Exception as e:
            print(f"Warning: Failed to parse PDF {name}: {e}")
            
    # 3. Plain Text / Markdown Files
    else:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if content.strip():
                    docs.append({"content": content.strip(), "metadata": {"source": name}})
        except Exception as e:
            print(f"Warning: Failed to parse text file {name}: {e}")
            
    return docs

def load_documents(data_dir: str = "data") -> List[Dict[str, Any]]:
    """Loads all base reviews and uploaded documents, then chunks them."""
    raw_docs = []
    
    # 1. Always load base reviews dataset
    if os.path.exists("realistic_restaurant_reviews.csv"):
        df = pd.read_csv("realistic_restaurant_reviews.csv")
        for _, r in df.iterrows():
            raw_docs.append({
                "content": f"{r['Title']} - {r['Review']}",
                "metadata": {"source": "reviews.csv"}
            })

    # 2. Also load additional uploaded files from data/ directory
    if os.path.exists(data_dir):
        for path in glob.glob(f"{data_dir}/*"):
            raw_docs.extend(parse_file_to_documents(path))

    # 3. Chunk documents into overlapping segments (pure Python)
    return split_documents(raw_docs, chunk_size=800, chunk_overlap=150)

# Initialize global Vector Store instance
vector_store = PureVectorStore()

# If the database file is missing or empty, build from documents
if not vector_store.documents:
    print("Building initial vector index from documents...")
    initial_chunks = load_documents()
    vector_store.add_documents(initial_chunks)

def ingest_new_file(path: str):
    """Parses, chunks, and indexes a single new file into the vector store."""
    raw_docs = parse_file_to_documents(path)
    if not raw_docs:
        return None
    new_chunks = split_documents(raw_docs, chunk_size=800, chunk_overlap=150)
    if new_chunks:
        vector_store.add_documents(new_chunks)
        return vector_store
    return None

def reset_database():
    """Deletes uploaded files, wipes the vector database, and rebuilds defaults."""
    # 1. Clear uploaded files in data directory
    if os.path.exists("data"):
        for f in os.listdir("data"):
            file_path = os.path.join("data", f)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Warning: Failed to delete file {f}: {e}")
                
    # 2. Clear vector store
    vector_store.clear()
    
    # 3. Reload default reviews
    default_chunks = load_documents()
    vector_store.add_documents(default_chunks)
    return vector_store