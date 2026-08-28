import os
import glob
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

def load_documents(data_dir: str = "data") -> list:
    docs = []
    # 1. Load multi-format files from data/ directory
    if os.path.exists(data_dir):
        for path in glob.glob(f"{data_dir}/*"):
            if os.path.isdir(path):
                continue
            name = os.path.basename(path)
            
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
                    from langchain_community.document_loaders import PyPDFLoader
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

    # 2. Fallback to default CSV if data folder is missing or empty
    if not docs and os.path.exists("realistic_restaurant_reviews.csv"):
        df = pd.read_csv("realistic_restaurant_reviews.csv")
        for _, r in df.iterrows():
            docs.append(Document(page_content=f"{r['Title']} - {r['Review']}", metadata={"source": "reviews.csv"}))

    # 3. Chunk documents into overlapping segments
    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)
    return splitter.split_documents(docs)

docs = load_documents()

# Chroma Vector Store (Dense Semantic Search)
vector_store = Chroma.from_documents(docs, embeddings, persist_directory="./chrome_langchain_db")
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# BM25 Retriever (Sparse Keyword Search)
bm25_retriever = BM25Retriever.from_documents(docs, k=5)

# Hybrid Retriever combining BM25 + Vector Similarity
retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]
)