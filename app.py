import os
import streamlit as st
from vector import retriever, ingest_new_file, reset_database, vector_store
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Page Configuration
st.set_page_config(
    page_title="KSHRD Pizza Restaurant QA Assistant",
    page_icon="./image.png",
    layout="wide"
)

# App Title & Header
st.image("./image.png", width=100)
st.title("KSHRD Pizza Restaurant RAG Assistant")
st.markdown("Ask questions about customer reviews, menu items, or your own uploaded documents.")

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
    
    # Only ingest if the file is new or modified
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        with st.sidebar.spinner(f"Ingesting {uploaded_file.name}..."):
            updated_retriever = ingest_new_file(file_path)
            if updated_retriever is not None:
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
                    st.markdown(f"**Chunk {i+1} | Source: `{src['source']}`**")
                    st.markdown(f"_{src['content']}_")
                    st.markdown("---")

# User Input & Generation Loop
if user_input := st.chat_input("What would you like to know?"):
    # Display user query
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state["messages"].append({"role": "user", "content": user_input})
    
    # Retrieve relevant documents live from ChromaDB
    with st.spinner("Searching reviews and documents..."):
        try:
            live_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            context_chunks = live_retriever.invoke(user_input)
        except Exception as e:
            st.error(f"Retrieval Error: {e}")
            context_chunks = []
            
    sources = []
    reviews_text = ""
    for doc in context_chunks:
        src = doc.metadata.get("source", "Unknown")
        sources.append({"source": src, "content": doc.page_content})
        reviews_text += f"\n---\n{doc.page_content}"

    # Load local Ollama model via OpenAI compatible API
    try:
        model = ChatOpenAI(
            model="llama3.2",
            openai_api_key="ollama",
            openai_api_base="http://localhost:11434/v1"
        )
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful pizza restaurant assistant. Answer questions using these customer reviews and documents:\n\n{reviews}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])
        
        # Format chat history for LangChain prompt
        langchain_history = []
        for m in st.session_state["messages"][:-1]:  # Exclude current query
            if m["role"] == "user":
                langchain_history.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                langchain_history.append(AIMessage(content=m["content"]))
                
        chain = prompt_template | model
        
        # Stream response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response_container = [""]
            
            def stream_generator():
                for chunk in chain.stream({
                    "reviews": reviews_text,
                    "history": langchain_history,
                    "question": user_input
                }):
                    text = chunk.content if hasattr(chunk, "content") else str(chunk)
                    full_response_container[0] += text
                    yield text
                    
            response_placeholder.write_stream(stream_generator())
            full_response = full_response_container[0]
            
            # Show sources
            if sources:
                with st.expander("🔍 View Retrieved Sources"):
                    for i, src in enumerate(sources):
                        st.markdown(f"**Chunk {i+1} | Source: `{src['source']}`**")
                        st.markdown(f"_{src['content']}_")
                        st.markdown("---")
                        
        st.session_state["messages"].append({
            "role": "assistant",
            "content": full_response,
            "sources": sources
        })
    except Exception as e:
        st.error(f"Error connecting to Ollama LLM: {e}")
        st.info("Please make sure Ollama is running locally and model 'llama3.2' is pulled.")
