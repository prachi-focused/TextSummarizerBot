# Simple document retriever with TF-IDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List

class DocumentRetriever:
    def __init__(self):
        # Setup text splitter and TF-IDF vectorizer
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.documents = []
        self.vectors = None
    
    def create_vector_store(self, content: str):
        """Split content and create TF-IDF vectors"""
        chunks = self.text_splitter.split_text(content)
        # Convert chunks to LangChain Document objects
        self.documents = [Document(page_content=chunk) for chunk in chunks]
        # Create TF-IDF vectors for all chunks
        self.vectors = self.vectorizer.fit_transform(chunks)
    
    def retrieve(self, query: str) -> List[Document]:
        """Get relevant documents using TF-IDF similarity"""
        if not self.documents or self.vectors is None:
            return []
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarity with all chunks
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        
        # Get top 3 most similar chunks
        top_indices = np.argsort(similarities)[-3:][::-1]
        relevant_docs = [self.documents[i] for i in top_indices if similarities[i] > 0]
        
        return relevant_docs[:3] if relevant_docs else []