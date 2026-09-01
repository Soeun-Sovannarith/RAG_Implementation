from vector import vector_store
from ollama_client import stream_chat

SYSTEM_INSTRUCTION = (
    "You are a helpful pizza restaurant assistant. "
    "Answer questions accurately using these customer reviews and documents:\n\n{reviews}"
)

def main():
    print("Pizza Restaurant RAG QA Assistant (Pure Python / No Frameworks)")
    print("Type 'q' or 'exit' to quit.")
    print("-" * 65)

    history = []  # Stores conversation turns: [{"role": "user"|"assistant", "content": "..."}]

    while True:
        try:
            q = input("\nUser: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not q or q.lower() in ["q", "quit", "exit"]:
            print("Goodbye!")
            break

        # 1. RETRIEVE: Pure Python Cosine Similarity Search on Chroma / Vector Store
        results = vector_store.search(q, k=5)
        reviews_context = "\n---\n".join([doc["content"] for doc in results])

        # 2. AUGMENT: Construct the prompt messages payload
        system_prompt = SYSTEM_INSTRUCTION.format(reviews=reviews_context)
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add past multi-turn history
        messages.extend(history)
        
        # Add current user query
        messages.append({"role": "user", "content": q})

        # 3. GENERATE: Stream tokens in real-time from Ollama /api/chat
        print("\nAssistant: ", end="", flush=True)
        full_response = ""
        
        for token in stream_chat(messages, model="llama3.2"):
            print(token, end="", flush=True)
            full_response += token
            
        print()  # Newline after streaming completes

        # 4. MEMORY: Save interaction for multi-turn chat context
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()