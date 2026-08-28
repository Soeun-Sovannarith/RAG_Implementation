# Gemini Agent Instructions

## Automatic Sub-Agent Delegation Rules

- **Skill Activation**: Automatically trigger the `auto-subagent` skill (`.agents/skills/auto-subagent/SKILL.md`) whenever an incoming task matches sub-agent delegation criteria.
- **Criteria for Auto Delegation**:
  - Web UI interaction / browsing -> Use `browser_subagent`.
  - Complex multi-step coding or refactoring -> Deconstruct into modular sub-tasks.
  - Background processes, data pipeline runs, or log monitoring -> Run asynchronously with sub-agent monitoring.

---

## Fast Architectural Index (Token-Efficient Context)

AI assistants MUST reference this index to understand project architecture instantly without spending tokens re-indexing files:

- **Project**: Local RAG QA System for Pizza Restaurant Reviews (`LocalAIAgentWithRAG`).
- **Core Pipeline**: CSV (`realistic_restaurant_reviews.csv`) -> `Document` Objects -> `mxbai-embed-large` (Ollama Embeddings) -> Chroma Vector Store (`./chrome_langchain_db`) -> `llama3.2` (Ollama LLM) via LangChain prompt pipeline.
- **Key Files**:
  - [main.py](file:///Users/sovannarith/Desktop/LocalAIAgentWithRAG/main.py): CLI interactive query REPL loop.
  - [vector.py](file:///Users/sovannarith/Desktop/LocalAIAgentWithRAG/vector.py): Data loader, Chroma DB creation, and retriever exporter ($k=5$).
  - [realistic_restaurant_reviews.csv](file:///Users/sovannarith/Desktop/LocalAIAgentWithRAG/realistic_restaurant_reviews.csv): Source dataset (124 reviews).
- **Execution Command**: `python main.py`
