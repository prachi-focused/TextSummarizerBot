# Simple Text Summarizer Bot
from rag_chain import RAGChain

def main():
    """Main Summarizer Bot"""

    # Initialize RAG Chain
    rag_chain = RAGChain()
    
    print("🤖 Welcome to the Text Summarizer Bot!")
    print("I can summarize content from URLs and answer questions about it.")
    print("-" * 60)
    
    # Get URL from user
    while True:
        url = input("\n📎 Please enter a URL to summarize: ").strip()
        if not url:
            print("Please enter a valid URL.")
            continue
        
        # Process URL and get summary
        print("\n🔄 Processing URL...")
        summary = rag_chain.process_input(url)
        
        if "Failed" in summary or "No relevant content" in summary:
            print(f"❌ Error: {summary}")
            continue
        else:
            print(f"\n📋 Summary:\n{summary}")
            break
    
    # Question loop
    print("\n" + "=" * 60)
    print("✅ Content processed! Now you can ask questions about it.")
    print("Type 'bye' or 'exit' to quit.")
    print("=" * 60)
    
    while True:
        question = input("\n❓ Your question: ").strip()
        
        # Check for exit commands
        if question.lower() in ['bye', 'exit', 'quit']:
            print("\n👋 Goodbye! Thanks for using the Text Summarizer Bot!")
            break
        
        if not question:
            print("Please enter a question.")
            continue
        
        # Get answer
        answer = rag_chain.process_input(question)
        print(f"\n💡 Answer: {answer}")

if __name__ == "__main__":
    main()