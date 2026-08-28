from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from vector import retriever

model = OllamaLLM(model="llama3.2")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful pizza restaurant assistant. Answer questions using these customer reviews:\n\n{reviews}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

chain = prompt | model
history = []

print("Pizza Restaurant RAG QA Assistant (Type 'q' to quit)")
print("-" * 50)

while True:
    q = input("\nUser: ").strip()
    if not q or q.lower() in ["q", "quit", "exit"]:
        print("Goodbye!")
        break

    # Retrieve relevant review chunks (Hybrid BM25 + Vector Search)
    context_chunks = retriever.invoke(q)
    reviews = "\n---\n".join([d.page_content for d in context_chunks])

    # Stream answer in real-time as tokens arrive
    print("\nAssistant: ", end="", flush=True)
    full_response = ""
    
    for chunk in chain.stream({"reviews": reviews, "history": history, "question": q}):
        print(chunk, end="", flush=True)
        full_response += chunk
        
    print()  # Newline after streaming completes

    # Track conversation history
    history.extend([HumanMessage(content=q), AIMessage(content=full_response)])