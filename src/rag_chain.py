# RAG Chain - handles URL input or vector store queries
from retriever import DocumentRetriever
from url_fetcher import fetch_url_content_as_chunks, validate_url
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
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
                                                       
            Context: {context}
            Question: {question}
        """)
        
        # Create the retrieval chain using LCEL
        self.rag_chain = None
   
    def process_url(self, url: str) -> str:
        """Fetch content from URL and initialize vector store"""
        # Fetch content
        content = fetch_url_content_as_chunks(url)
        if not content:
            return "Failed to fetch URL content"
        
        # Create vector store
        self.retriever.create_vector_store(content)
        
        # Initialize the RAG chain using LCEL pattern
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        def retrieve_and_format(question):
            docs = self.retriever.retrieve(question)
            return format_docs(docs)
        
        self.rag_chain = (
            {"context": RunnableLambda(retrieve_and_format), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Return confirmation
        return "Vector store initialized."
    
    def query_data(self, question: str) -> str:
        """Answer question from existing vector store using LangChain retrieval chain"""
        if not self.rag_chain:
            return "No vector store initialized. Please process a URL first."
        
        try:
            # Use the LangChain retrieval chain
            return self.rag_chain.invoke(question)
        except Exception as e:
            return f"Error answering question: {e}"