import os
import glob
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.document_loaders import PyPDFLoader

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

def parse_file_to_documents(path: str) -> list:
    """Parses a file based on its extension and returns a list of Document objects."""
    docs = []
    name = os.path.basename(path)
    if os.path.isdir(path):
        return docs
        
    # CSV structured files
    if path.endswith(".csv"):
        try:
            df = pd.read_csv(path)
            for _, r in df.iterrows():
                text = f"{r.get('Title', '')} - {r.get('Review', r.get('text', str(r.to_dict())))}"
                docs.append(Document(page_content=text.strip("- "), metadata={"source": name}))
        except Exception as e:
            print(f"Warning: Failed to parse CSV {name}: {e}")
            
    # PDF documents
    elif path.endswith(".pdf"):
        try:
            pdf_loader = PyPDFLoader(path)
            pdf_docs = pdf_loader.load()
            for d in pdf_docs:
                d.metadata["source"] = name
            docs.extend(pdf_docs)
        except Exception as e:
            print(f"Warning: Could not read PDF {name}: {e}")
            
    # Any other text-based extension (.txt, .md, .json, .log, .html, etc.)
    else:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if content.strip():
                    docs.append(Document(page_content=content, metadata={"source": name}))
        except Exception as e:
            print(f"Warning: Could not read text file {name}: {e}")
            
    return docs

def load_documents(data_dir: str = "data") -> list:
    docs = []
    # 1. Load multi-format files from data/ directory
    if os.path.exists(data_dir):
        for path in glob.glob(f"{data_dir}/*"):
            docs.extend(parse_file_to_documents(path))

    # 2. Fallback to default CSV if data folder is missing or empty
    if not docs and os.path.exists("realistic_restaurant_reviews.csv"):
        df = pd.read_csv("realistic_restaurant_reviews.csv")
        for _, r in df.iterrows():
            docs.append(Document(page_content=f"{r['Title']} - {r['Review']}", metadata={"source": "reviews.csv"}))

    # 3. Chunk documents into overlapping segments
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    return splitter.split_documents(docs)

def ingest_new_file(path: str):
    """Parses, chunks, and indexes a single new file, updating the retrievers."""
    raw_docs = parse_file_to_documents(path)
    if not raw_docs:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    new_chunks = splitter.split_documents(raw_docs)
    if new_chunks:
        return update_retriever_with_new_docs(new_chunks)
    return None

docs = load_documents()

# Chroma Vector Store (Dense Semantic Search)
# Initialize Chroma: if the directory exists and is not empty, load from disk. Otherwise index the initial docs.
if os.path.exists("./chrome_langchain_db") and os.listdir("./chrome_langchain_db"):
    vector_store = Chroma(persist_directory="./chrome_langchain_db", embedding_function=embeddings)
else:
    vector_store = Chroma.from_documents(docs, embeddings, persist_directory="./chrome_langchain_db")

vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# BM25 Retriever (Sparse Keyword Search)
bm25_retriever = BM25Retriever.from_documents(docs, k=5)

# Hybrid Retriever combining BM25 + Vector Similarity
retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]
)

def reset_database():
    """Deletes all persistent files/embeddings and rebuilds the database to defaults."""
    global vector_store, retriever, bm25_retriever, vector_retriever, docs
    
    # 1. Clear Chroma vector store documents using API (keeps file handles valid)
    try:
        all_ids = vector_store.get()["ids"]
        if all_ids:
            vector_store.delete(ids=all_ids)
    except Exception as e:
        print(f"Warning: Failed to clear Chroma collection: {e}")
        
    # 2. Clear uploaded files in data directory
    if os.path.exists("data"):
        for f in os.listdir("data"):
            file_path = os.path.join("data", f)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Warning: Failed to delete file {f}: {e}")
        
    # 3. Reload base documents & add them back to Chroma
    docs = load_documents()
    if docs:
        vector_store.add_documents(docs)
        
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    bm25_retriever = BM25Retriever.from_documents(docs, k=5)
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]
    )
    return retriever

def update_retriever_with_new_docs(new_docs_chunks):
    """Adds new chunks to vector store and updates both BM25 and Ensemble retrievers."""
    global vector_store, retriever, bm25_retriever, vector_retriever, docs
    
    # 1. Add to Chroma
    vector_store.add_documents(new_docs_chunks)
    
    # 2. Reload all documents from disk to rebuild BM25 (so it matches Chroma)
    docs = load_documents()
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    bm25_retriever = BM25Retriever.from_documents(docs, k=5)
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]
    )
    return retriever