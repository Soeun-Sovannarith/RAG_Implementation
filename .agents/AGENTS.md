# Project Agent Directives

## Automatic Sub-Agent Delegation

- **Automatic Activation**: Whenever a user prompt involves browser interaction, multi-step web automation, modular feature development, deep research, or concurrent background tasks, automatically load and apply the `auto-subagent` skill (`.agents/skills/auto-subagent/SKILL.md`).
- **Autonomous Delegation**: Do not wait for manual user requests to spawn sub-agents when a task clearly benefits from delegated parallel execution or specialized sub-agent tools (e.g. `browser_subagent`).
- **Synthesis & Verification**: Ensure all sub-agent tasks produce verifiable outputs that are integrated and validated before completing the main request.

---

## Project Context & Fast Architectural Index (Token Optimization)

This section provides AI assistants with instant, token-efficient context regarding the **LocalAIAgentWithRAG** project without needing to re-read source files.

### 1. Project Summary
- **Name**: LocalAIAgentWithRAG
- **Description**: A local Retrieval-Augmented Generation (RAG) QA system built with LangChain, Ollama, ChromaDB, and Pandas to answer questions based on restaurant reviews.
- **Language / Runtime**: Python 3.14 (managed via Pipenv or `requirements.txt`).

### 2. Architecture & File Mapping
- [main.py](file:///Users/sovannarith/Desktop/LocalAIAgentWithRAG/main.py): CLI interactive REPL loop. Loads Ollama LLM (`llama3.2`), imports `retriever` from `vector.py`, constructs `ChatPromptTemplate`, retrieves top $k=5$ reviews per prompt, and outputs response.
- [vector.py](file:///Users/sovannarith/Desktop/LocalAIAgentWithRAG/vector.py): Data ingestion & vector retriever setup. Loads `realistic_restaurant_reviews.csv`, checks if `./chrome_langchain_db` exists (creates & adds embedded documents if missing), uses `mxbai-embed-large` via `OllamaEmbeddings`, and exports Chroma retriever (`k=5`).
- [realistic_restaurant_reviews.csv](file:///Users/sovannarith/Desktop/LocalAIAgentWithRAG/realistic_restaurant_reviews.csv): Dataset containing 124 restaurant reviews with columns: `Title`, `Date`, `Rating`, `Review`.
- [chrome_langchain_db](file:///Users/sovannarith/Desktop/LocalAIAgentWithRAG/chrome_langchain_db): Local Chroma vector database persistence folder (`collection_name="restaurant_reviews"`).
- [requirements.txt](file:///Users/sovannarith/Desktop/LocalAIAgentWithRAG/requirements.txt) & [Pipfile](file:///Users/sovannarith/Desktop/LocalAIAgentWithRAG/Pipfile): Core dependencies (`langchain`, `langchain-ollama`, `langchain-chroma`, `pandas`).

### 3. Models & Settings
- **LLM**: `OllamaLLM(model="llama3.2")`
- **Embeddings**: `OllamaEmbeddings(model="mxbai-embed-large")`
- **Vector Search**: ChromaDB similarity retriever (`search_kwargs={"k": 5}`)

### 4. Running the Project
```bash
python main.py
```
