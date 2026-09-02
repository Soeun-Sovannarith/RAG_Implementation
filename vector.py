import os
import glob
import math
import pandas as pd
import chromadb
from typing import List, Dict, Any
from pypdf import PdfReader
from splitter import split_documents
from ollama_client import get_embedding

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "restaurant_reviews"

class ChromaVectorStore:
    """
    Vector database backed by ChromaDB with HNSW (Hierarchical Navigable Small World) index for ANN search.
    """
    def __init__(self, db_path: str = CHROMA_DB_PATH):
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=self.db_path)
        # Configure HNSW with Cosine Similarity space
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, docs: List[Dict[str, Any]]):
        """Generates embeddings for documents and appends them to ChromaDB HNSW collection."""
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        current_count = self.collection.count()
        
        for i, doc in enumerate(docs):
            text = doc.get("content", "").strip()
            if not text:
                continue
            emb = get_embedding(text)
            if emb:
                doc_id = f"doc_{current_count + i + 1}_{abs(hash(text))}"
                ids.append(doc_id)
                embeddings.append(emb)
                documents.append(text)
                metadatas.append(doc.get("metadata", {}))
                
        if ids:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Embeds query and performs HNSW Approximate Nearest Neighbor (ANN) search in ChromaDB.
        """
        total_count = self.collection.count()
        if total_count == 0:
            return []
            
        query_emb = get_embedding(query)
        if not query_emb:
            return []
            
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=min(k, total_count)
        )
        
        scored_docs = []
        if results and results.get("documents") and results["documents"][0]:
            docs_list = results["documents"][0]
            meta_list = results.get("metadatas", [[]])[0]
            dist_list = results.get("distances", [[]])[0]
            
            for doc_text, meta, dist in zip(docs_list, meta_list, dist_list):
                # In ChromaDB cosine distance d = 1 - cosine_similarity
                score = 1.0 - dist if dist is not None else 0.0
                scored_docs.append({
                    "content": doc_text,
                    "metadata": meta if meta is not None else {},
                    "score": score
                })
                
        return scored_docs

    def clear(self):
        """Clears all records from ChromaDB collection."""
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

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

# Initialize global Chroma Vector Store instance
vector_store = ChromaVectorStore()

# If the database collection is empty, build initial vector index from documents
if vector_store.collection.count() == 0:
    print("Building initial ChromaDB HNSW vector index from documents...")
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