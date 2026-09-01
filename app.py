import os
import streamlit as st
from vector import vector_store, ingest_new_file, reset_database
from ollama_client import stream_chat

# Page Configuration
st.set_page_config(
    page_title="KSHRD Pizza Restaurant QA Assistant",
    page_icon="./image.png",
    layout="wide"
)

# App Title & Header
st.image("./image.png", width=100)
st.title("KSHRD Pizza Restaurant RAG Assistant")
st.markdown("Ask questions about customer reviews, menu items, or your own uploaded documents (Pure Python / No LangChain).")

# Initialize chat message history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Sidebar Controls
st.sidebar.header("⚙️ Document Management")

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload new reviews or restaurant documentation", 
    type=["csv", "pdf", "txt", "md"]
)

if uploaded_file is not None:
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", uploaded_file.name)
    
    # Ingest new file if it's not already uploaded or if newly selected
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        with st.sidebar.spinner(f"Ingesting {uploaded_file.name}..."):
            res = ingest_new_file(file_path)
            if res is not None:
                st.sidebar.success(f"Successfully indexed: {uploaded_file.name}")
                st.rerun()
            else:
                st.sidebar.error(f"Failed to process {uploaded_file.name}")

# List current sources
st.sidebar.markdown("### Indexed Sources")
st.sidebar.markdown("- `realistic_restaurant_reviews.csv` (Default Reviews)")
if os.path.exists("data"):
    uploaded_docs = [f for f in os.listdir("data") if not f.startswith(".")]
    for f in sorted(uploaded_docs):
        st.sidebar.markdown(f"- `{f}`")

st.sidebar.markdown("---")

# Reset Database Action
if st.sidebar.button("Reset Database to Defaults", type="primary"):
    with st.sidebar.spinner("Resetting database..."):
        reset_database()
        st.session_state["messages"] = []
        st.sidebar.success("Database has been reset to default reviews!")
        st.rerun()

# Display chat messages from history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("View Retrieved Sources"):
                for i, src in enumerate(msg["sources"]):
                    score_info = f" | Similarity Score: {src['score']:.4f}" if "score" in src else ""
                    st.markdown(f"**Chunk {i+1} | Source: `{src['source']}`{score_info}**")
                    st.markdown(f"_{src['content']}_")
                    st.markdown("---")

# User Input & Generation Loop
if user_input := st.chat_input("What would you like to know?"):
    # 1. Display user query
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state["messages"].append({"role": "user", "content": user_input})
    
    # 2. Retrieve relevant documents using pure Python Cosine Similarity search
    with st.spinner("Searching reviews and documents..."):
        try:
            results = vector_store.search(user_input, k=5)
        except Exception as e:
            st.error(f"Retrieval Error: {e}")
            results = []
            
    sources = []
    reviews_text = ""
    for doc in results:
        src = doc.get("metadata", {}).get("source", "Unknown")
        score = doc.get("score", 0.0)
        sources.append({"source": src, "content": doc["content"], "score": score})
        reviews_text += f"\n---\n{doc['content']}"

    # 3. Format Prompt & Chat History
    system_prompt = (
        "You are a helpful pizza restaurant assistant. "
        f"Answer questions using these customer reviews and documents:\n\n{reviews_text}"
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    for m in st.session_state["messages"][:-1]:  # Exclude current query
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_input})
    
    # 4. Stream response from Ollama
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response_container = [""]
        
        def stream_generator():
            for token in stream_chat(messages, model="llama3.2"):
                full_response_container[0] += token
                yield token
                
        response_placeholder.write_stream(stream_generator())
        full_response = full_response_container[0]
        
        # Show source evidence
        if sources:
            with st.expander("View Retrieved Sources"):
                for i, src in enumerate(sources):
                    st.markdown(f"**Chunk {i+1} | Source: `{src['source']}` (Score: {src['score']:.4f})**")
                    st.markdown(f"_{src['content']}_")
                    st.markdown("---")
                    
    st.session_state["messages"].append({
        "role": "assistant",
        "content": full_response,
        "sources": sources
    })
