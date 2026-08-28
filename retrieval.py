from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from vector import load_review_documents, vector_store

def get_hybrid_retriever(k: int = 5, bm25_weight: float = 0.4, vector_weight: float = 0.6):
    """
    Creates an EnsembleRetriever combining BM25 keyword search and Dense Vector Similarity search.
    """
    # 1. Load documents for BM25 Sparse Keyword Retriever
    documents = load_review_documents()
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k

    # 2. Get Chroma Dense Vector Retriever
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # 3. Create Ensemble (Hybrid) Retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight]
    )
    return ensemble_retriever

# Export default hybrid retriever for usage across app
hybrid_retriever = get_hybrid_retriever(k=5)

if __name__ == "__main__":
    print("--- Testing Hybrid (BM25 + Vector) Retriever ---")
    query = "crispy crust garlic knots"
    docs = hybrid_retriever.invoke(query)
    print(f"\nQuery: '{query}' returned {len(docs)} hybrid chunks:")
    for i, doc in enumerate(docs):
        print(f"\nResult {i+1} (Source: {doc.metadata.get('source')}, Rating: {doc.metadata.get('rating')} stars):")
        print(f"Content: {doc.page_content}")
