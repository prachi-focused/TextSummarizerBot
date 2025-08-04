# Import relevant functionality
from dotenv import load_dotenv
from rag_chain import RAGChain

# Load environment variables from .env file
load_dotenv()

# Initialize RAG chain
rag_chain = RAGChain()

def process_url(url: str) -> str:
    """Process a URL and initialize the vector store with its content."""
    try:
        status = rag_chain.process_input(url)
        return status
    except Exception as e:
        error_msg = f"Error processing URL: {e}"
        return error_msg

def ask_question(question: str) -> str:
    """Ask a question about the processed content using RAG."""
    print("-" * 50)
    print(f"Question: {question}")
    print("-" * 30)
    
    try:
        # Use RAG chain to answer question
        answer = rag_chain.process_input(question)
        print("Answer:")
        print(answer)
        return answer
    except Exception as e:
        error_msg = f"Error answering question: {e}"
        print(f"❌ {error_msg}")
        return error_msg

if __name__ == "__main__":
    # Example usage
    url = "https://www.smithsonianmag.com/smart-news/24-billion-gallons-of-water-burst-through-greenlands-ice-sheet-from-a-hidden-lake-in-2014-scientists-just-pieced-together-what-happened-180987085/?utm_source=firefox-newtab-en-us"
    
    # Process URL and set up vector store
    process_url(url)
    
    # Ask questions about the content
    questions = [
        "Summarize the content in 3 simple sentences",
        "What effect will this have on the sea levels?"
    ]
    
    for question in questions:
        ask_question(question) 