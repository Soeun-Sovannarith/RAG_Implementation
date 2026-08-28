from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from typing import List

def get_text_splitter(chunk_size: int = 300, chunk_overlap: int = 50) -> RecursiveCharacterTextSplitter:
    """
    Returns a configured RecursiveCharacterTextSplitter.
    Attempts to split on ["\n\n", "\n", ". ", " ", ""] to keep semantic blocks together.
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
    return splitter.split_documents(documents)

def split_markdown_by_headers(markdown_text: str) -> List[Document]:
    """
    Demonstrates Document-Structure-Aware Splitting by splitting markdown text on headers (#, ##).
    Automatically attaches header titles into metadata for precise section retrieval.
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    return markdown_splitter.split_text(markdown_text)

if __name__ == "__main__":
    print("=== 1. Testing Recursive Character Text Splitter ===")
    sample_text = (
        "The pizza at Mario's Pizzeria was absolutely magnificent! "
        "The crust was crispy on the outside, light and airy on the inside. "
        "The tomato sauce had just the right balance of sweet and tangy flavors."
    )
    doc = Document(page_content=sample_text, metadata={"source": "sample_review.txt", "rating": 5})
    chunks = split_documents([doc], chunk_size=150, chunk_overlap=30)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} [{len(chunk.page_content)} chars]: {chunk.page_content}")

    print("\n=== 2. Testing Structure-Aware Markdown Header Splitter ===")
    sample_md = """# Pizza Menu Options

## Signature Pizzas
Our signature pizzas are crafted using 72-hour fermented sourdough crust.

## Specialty Drinks
We offer authentic Italian sodas, house sangria, and local craft IPAs.
"""
    md_chunks = split_markdown_by_headers(sample_md)
    for i, chunk in enumerate(md_chunks):
        print(f"MD Chunk {i+1} [Metadata: {chunk.metadata}]: {chunk.page_content}")
