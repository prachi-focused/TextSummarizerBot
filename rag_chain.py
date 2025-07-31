# RAG Chain - handles URL input or vector store queries
from retriever import DocumentRetriever
from generator import TextGenerator
from url_fetcher import fetch_url_content, validate_url

class RAGChain:
    def __init__(self):
        # Initialize components
        self.retriever = DocumentRetriever()
        self.generator = TextGenerator()
   
    def process_input(self, user_input: str) -> str:
        """Check if input has URL, fetch data or retrieve from vector store"""
        # Check if input contains URL
        if validate_url(user_input):
            return self._process_url(user_input)
        else:
            return self._query_vector_store(user_input)
    
    def _process_url(self, url: str) -> str:
        """Fetch content from URL and create summary"""
        # Fetch content
        content = fetch_url_content(url)
        if not content:
            return "Failed to fetch URL content"
        
        # Create vector store
        self.retriever.create_vector_store(content)
        
        # Get summary
        context = self.retriever.get_context("Summarize this content")
        return self.generator.generate_response(context, "Summarize")
    
    def _query_vector_store(self, question: str) -> str:
        """Answer question from existing vector store"""
        context = self.retriever.get_context(question)
        if not context:
            return "No relevant content found"
        return self.generator.generate_response(context, question)