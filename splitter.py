from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def get_text_splitter(chunk_size: int = 300, chunk_overlap: int = 50) -> RecursiveCharacterTextSplitter:
    """
    Returns a configured RecursiveCharacterTextSplitter.
    RecursiveCharacterTextSplitter attempts to split on ["\n\n", "\n", " ", ""]
    to keep semantic blocks (paragraphs, sentences) together.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

def split_documents(documents: List[Document], chunk_size: int = 300, chunk_overlap: int = 50) -> List[Document]:
    """
    Splits a list of Document objects into smaller chunks while preserving metadata.
    """
    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.split_documents(documents)
    return split_docs

if __name__ == "__main__":
    # Example demonstration of text splitting strategy
    sample_text = (
        "The pizza at Mario's Pizzeria was absolutely magnificent! "
        "The crust was crispy on the outside, light and airy on the inside. "
        "The tomato sauce had just the right balance of sweet and tangy flavors. "
        "Service was fast and friendly, though the seating area can get quite crowded on Friday nights. "
        "We also tried the garlic knots, which were dripping in butter and fresh herbs. Highly recommended!"
    )
    
    doc = Document(page_content=sample_text, metadata={"source": "sample_review.txt", "rating": 5})
    
    print("--- Original Document ---")
    print(f"Length: {len(sample_text)} characters")
    
    chunks = split_documents([doc], chunk_size=150, chunk_overlap=30)
    
    print(f"\n--- Split Chunks (Total: {len(chunks)}) ---")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} [{len(chunk.page_content)} chars]: {chunk.page_content}")
