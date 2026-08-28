# Master RAG System Implementation Plan

This plan outlines the step-by-step tasks required to evolve the current single-turn baseline RAG project into a complete, production-grade RAG QA system covering all core concepts: text chunking, local embeddings, multi-vector DB setup (ChromaDB & Qdrant), hybrid retrieval, and conversational memory.

## User Review Required

> [!IMPORTANT]
> - **Vector Database Options**: We will add support for **Qdrant** alongside **ChromaDB** using an in-memory/local Qdrant setup. This requires installing `langchain-qdrant` and `qdrant-client`.
> - **Dependencies**: New requirements will be added to `requirements.txt` (`langchain-community`, `langchain-text-splitters`, `langchain-qdrant`, `qdrant-client`, `rank_bm25`).

## Proposed Tasks & Changes

---

### Task 1: Ingestion & Text Splitting Strategies (Chunking & Local Embeddings)
- **Goal**: Cover text splitting strategies (`RecursiveCharacterTextSplitter`, `SemanticChunker`) and multi-document loading (`.csv`, `.txt`, `.pdf`).
- **Files**:
  - [NEW] [splitter.py](file:///Users/sovannarith/Desktop/LocalAIAgentWithRAG/splitter.py): Text chunking module demonstrating fixed-size character splitting vs. semantic splitting.
  - [MODIFY] [requirements.txt](file:///Users/sovannarith/Desktop/LocalAIAgentWithRAG/requirements.txt): Add `langchain-text-splitters` and `langchain-community`.

---

### Task 2: Multi-Vector DB Setup (ChromaDB & Qdrant Integration)
- **Goal**: Implement vector store abstractions allowing seamless switching between **ChromaDB** and **Qdrant** with metadata filtering.
- **Files**:
  - [MODIFY] [vector.py](file:///Users/sovannarith/Desktop/LocalAIAgentWithRAG/vector.py): Modularize vector database creation so the user can select backend (`"chroma"` or `"qdrant"`) and apply metadata filters (e.g. rating >= 4).

---

### Task 3: Advanced Retrieval Pipeline (Hybrid Search & Re-ranking)
- **Goal**: Combine dense vector retrieval (`mxbai-embed-large`) with sparse keyword search (`BM25Retriever`) using an `EnsembleRetriever`.
- **Files**:
  - [NEW] [retrieval.py](file:///Users/sovannarith/Desktop/LocalAIAgentWithRAG/retrieval.py): Advanced retrieval pipeline combining BM25 keyword matching and vector similarity.

---

### Task 4: Complete "Chat with Documents" App with Conversation Memory
- **Goal**: Turn the CLI REPL into a full multi-turn conversational chat system that remembers previous user questions and answers.
- **Files**:
  - [MODIFY] [main.py](file:///Users/sovannarith/Desktop/LocalAIAgentWithRAG/main.py): Upgrade REPL loop to include `RunnableWithMessageHistory` / `ChatMessageHistory`.

---

## Verification Plan

### Automated / Scripted Verification
- Run `python vector.py` with `db_type="chroma"` and `db_type="qdrant"` to verify both vector databases create indexes cleanly.
- Run `python splitter.py` to inspect chunking output lengths and token counts.
- Run `python retrieval.py` to verify hybrid BM25 + Vector retrieval returns top relevant results.

### Manual Verification
- Run `python main.py` and ask follow-up questions (e.g., *"What is the top rated pizza?"* followed by *"How much does it cost?"*) to confirm context memory works across turns.
