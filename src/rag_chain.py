# RAG Chain - handles URL input or vector store queries
from retriever import DocumentRetriever
from url_fetcher import fetch_url_content_as_chunks, validate_url
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

class RAGChain:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize components
        self.retriever = DocumentRetriever()
        
        # Initialize LLM and prompt
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
        self.prompt = ChatPromptTemplate.from_template("""
            You are a helpful assistant that answers questions based on the context. 
            Stick to the context for the answer.
            If the question is not related to the context, say "I don't know" or "I don't have information about that".
                                                       
            Context: {context}
            Question: {question}
        """)
        
   
    def process_url(self, url: str) -> str:
        """Fetch content from URL and initialize vector store"""
        # Fetch content
        content = fetch_url_content_as_chunks(url)
        if not content:
            return "Failed to fetch URL content"
        
        # Create vector store
        self.retriever.create_vector_store(content)
        
        # Initialize the RAG chain using LCEL pattern
        
        # Return confirmation
        return "Vector store initialized."
    
    def query_data_with_context(self, question: str) -> dict:
        """Answer question and return both answer and retrieved context"""
        
        try:
            # Retrieve documents
            docs = self.retriever.retrieve(question)
            context = "\n\n".join(doc.page_content for doc in docs)
            
            # Use the retrieved context directly with prompt and LLM
            prompt_input = {"context": context, "question": question}
            formatted_prompt = self.prompt.format(**prompt_input)
            answer = self.llm.invoke(formatted_prompt).content
            
            return {
                "answer": answer,
                "context": context
            }
        except Exception as e:
            return {
                "answer": f"Error answering question: {e}",
                "context": ""
            }
    
    def query_data(self, question: str) -> str:
        """Answer question from existing vector store using LangChain retrieval chain"""
        # Use the main method and extract just the answer
        result = self.query_data_with_context(question)
        return result["answer"]