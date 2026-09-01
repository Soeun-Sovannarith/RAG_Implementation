from vector import vector_store

def get_vector_retriever(k: int = 5):
    """
    Returns a Chroma Dense Vector Similarity Retriever for top-k nearest document chunks.
    """
    return vector_store.as_retriever(search_kwargs={"k": k})

# Export default vector retriever
vector_retriever = get_vector_retriever(k=5)

if __name__ == "__main__":
    print("--- Testing Chroma Vector Retriever ---")
    query = "crispy crust garlic knots"
    docs = vector_retriever.invoke(query)
    print(f"\nQuery: '{query}' returned {len(docs)} vector chunks:")
    for i, doc in enumerate(docs):
        src = doc.metadata.get('source', 'Unknown')
        print(f"\nResult {i+1} (Source: {src}):")
        print(f"Content: {doc.page_content}")
