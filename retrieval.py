from vector import vector_store

def search_documents(query: str, k: int = 5):
    """
    Performs pure Python Cosine Similarity search on the vector store.
    """
    return vector_store.search(query, k=k)

if __name__ == "__main__":
    print("--- Testing Pure Python Vector Retrieval ---")
    query = "crispy crust garlic knots"
    results = search_documents(query, k=5)
    print(f"\nQuery: '{query}' returned {len(results)} vector chunks:")
    for i, doc in enumerate(results):
        src = doc["metadata"].get("source", "Unknown")
        score = doc.get("score", 0.0)
        print(f"\nResult {i+1} (Score: {score:.4f} | Source: {src}):")
        print(f"Content: {doc['content']}")
