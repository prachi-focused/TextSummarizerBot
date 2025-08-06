# Import relevant functionality
from dotenv import load_dotenv
from rag_chain import RAGChain
from url_fetcher import validate_url

# Load environment variables from .env file
load_dotenv()

# Initialize RAG chain
rag_chain = RAGChain()

def process_url(url: str) -> str:
    """Process a URL and initialize the vector store with its content."""
    try:
        # Validate URL first
        if not validate_url(url):
            return "Invalid URL format. Please provide a valid HTTP/HTTPS URL."
        
        # Process the URL using RAG chain
        status = rag_chain.process_url(url)
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
        # Use RAG chain to query vector store
        answer = rag_chain.query_data(question)
        print("Answer:")
        print(answer)
        return answer
    except Exception as e:
        error_msg = f"Error answering question: {e}"
        print(f"❌ {error_msg}")
        return error_msg

if __name__ == "__main__":
    # Example usage
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    
    # Process URL and set up vector store
    process_url(url)
    
    # Ask questions about the content
    questions = [
        "Summarize the main concepts of artificial intelligence in 3 sentences",
        "How old is the field of artificial intelligence?",
    ]
    
    for question in questions:
        ask_question(question)