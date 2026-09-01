from typing import List, Dict, Any

def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[str]:
    """
    Pure Python text chunker with overlapping sliding window.
    Splits text by natural separators (\n\n, \n, . , space) to preserve context.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        
        # Try to find a natural break point (paragraph, sentence, word)
        chunk_slice = text[start:end]
        cut_index = -1
        
        for separator in ["\n\n", "\n", ". ", " "]:
            last_pos = chunk_slice.rfind(separator)
            if last_pos != -1 and last_pos > int(chunk_size * 0.4): # ensure chunk is not too small
                cut_index = last_pos + len(separator)
                break
        
        if cut_index == -1:
            cut_index = chunk_size
            
        chunk = text[start:start + cut_index].strip()
        if chunk:
            chunks.append(chunk)
            
        start += max(1, cut_index - chunk_overlap)
        
    return chunks

def split_documents(documents: List[Dict[str, Any]], chunk_size: int = 800, chunk_overlap: int = 150) -> List[Dict[str, Any]]:
    """
    Splits a list of document dicts {"content": str, "metadata": dict} into smaller chunks.
    """
    chunked_docs = []
    for doc in documents:
        content = doc.get("content", "")
        metadata = doc.get("metadata", {})
        chunks = split_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk in chunks:
            chunked_docs.append({
                "content": chunk,
                "metadata": metadata.copy()
            })
    return chunked_docs

if __name__ == "__main__":
    sample = (
        "The pizza at Mario's Pizzeria was absolutely magnificent! "
        "The crust was crispy on the outside, light and airy on the inside. "
        "The tomato sauce had just the right balance of sweet and tangy flavors."
    )
    chunks = split_text(sample, chunk_size=70, chunk_overlap=20)
    print("Testing pure Python text splitter:")
    for i, c in enumerate(chunks):
        print(f"Chunk {i+1} ({len(c)} chars): {c}")
