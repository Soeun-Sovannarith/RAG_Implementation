---
name: project-context
description: Fast architectural context and quick lookup rules for the LocalAIAgentWithRAG codebase.
---

# Project Context & Quick Lookup Skill

Use this skill to quickly understand the project architecture, dependencies, data models, and workflow without re-reading source code files.

## Project Quick Reference

| Component | Technology | File / Location | Key Details |
|---|---|---|---|
| **LLM Model** | Ollama (`llama3.2`) | [main.py](file:///Users/sovannarith/Desktop/LocalAIAgentWithRAG/main.py) | `OllamaLLM(model="llama3.2")` |
| **Embeddings** | Ollama (`mxbai-embed-large`) | [vector.py](file:///Users/sovannarith/Desktop/LocalAIAgentWithRAG/vector.py) | `OllamaEmbeddings(model="mxbai-embed-large")` |
| **Vector Store** | ChromaDB (`collection_name="restaurant_reviews"`) | [vector.py](file:///Users/sovannarith/Desktop/LocalAIAgentWithRAG/vector.py) | Saved at `./chrome_langchain_db` |
| **Retriever** | LangChain VectorStore Retriever | [vector.py](file:///Users/sovannarith/Desktop/LocalAIAgentWithRAG/vector.py) | `search_kwargs={"k": 5}` |
| **Dataset** | CSV (124 reviews) | [realistic_restaurant_reviews.csv](file:///Users/sovannarith/Desktop/LocalAIAgentWithRAG/realistic_restaurant_reviews.csv) | Columns: `Title`, `Date`, `Rating`, `Review` |
| **CLI App** | Python REPL | [main.py](file:///Users/sovannarith/Desktop/LocalAIAgentWithRAG/main.py) | Interactive query loop |

## Data Flow Diagram

```
[realistic_restaurant_reviews.csv]
              │
              ▼
    [vector.py: Document Creation]
    page_content = Title + Review
    metadata = {rating, date}
              │
              ▼
 [OllamaEmbeddings: mxbai-embed-large]
              │
              ▼
  [Chroma Store: ./chrome_langchain_db]
              │
              ▼ (Retriever k=5)
    [main.py: Question Input] ──► [OllamaLLM: llama3.2] ──► Answer
```

## Maintenance & Extension Guidelines

1. **Changing Models**:
   - To update the LLM, modify `model="<new_model>"` in [main.py](file:///Users/sovannarith/Desktop/LocalAIAgentWithRAG/main.py#L5).
   - To update embeddings, modify `model="<new_embedding>"` in [vector.py](file:///Users/sovannarith/Desktop/LocalAIAgentWithRAG/vector.py#L8) and delete `./chrome_langchain_db` to trigger re-indexing.
2. **Adding Metadata Filtering**:
   - Edit [vector.py](file:///Users/sovannarith/Desktop/LocalAIAgentWithRAG/vector.py#L35) to add filter parameters in `search_kwargs`.
