# Simple document retriever with TF-IDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class DocumentRetriever:
    def __init__(self):
        # Setup text splitter and TF-IDF vectorizer
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.chunks = []
        self.vectors = None
        print("Retriever ready")
    
    def create_vector_store(self, content: str):
        """Split content and create TF-IDF vectors"""
        self.chunks = self.text_splitter.split_text(content)
        # Create TF-IDF vectors for all chunks
        self.vectors = self.vectorizer.fit_transform(self.chunks)
        print(f"Created vector store with {len(self.chunks)} chunks")
    
    def get_context(self, query: str) -> str:
        """Get relevant context using TF-IDF similarity"""
        if not self.chunks or self.vectors is None:
            return ""
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarity with all chunks
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        
        # Get top 3 most similar chunks
        top_indices = np.argsort(similarities)[-3:][::-1]
        relevant_chunks = [self.chunks[i] for i in top_indices if similarities[i] > 0]
        
        return "\n".join(relevant_chunks[:3]) if relevant_chunks else ""