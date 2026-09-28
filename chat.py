import sys
from vector_db import VectorDB
from llm import generate_answer

def build_prompt(query: str, context_chunks: list[str]) -> str:
    context_str = "\n\n".join(context_chunks)
    prompt = (
        f"You are a helpful study assistant. Use the following context to answer the question.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    return prompt

def main():
    print("Initializing Vector DB...")
    db = VectorDB()
    
    print("Chatbot ready. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            query = input("\nYou: ")
            if query.lower() in ['exit', 'quit']:
                break
            
            print("Searching for context...")
            relevant_chunks = db.search(query, n_results=3)
            
            if not relevant_chunks:
                print("No relevant context found.")
                continue
                
            prompt = build_prompt(query, relevant_chunks)
            
            print("\nAssistant: ", end="", flush=True)
            generate_answer(prompt)
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
