"""
Dataset Setup for Text Summarizer Evaluation
This file creates a LangSmith dataset for evaluating RAG-based Q&A.
"""

from langsmith import Client
import os
from dotenv import load_dotenv

load_dotenv()


def create_summarizer_dataset():
    """Create a dataset with URL + question pairs for summarizer evaluation."""
    
    client = Client()

    # Programmatically create a dataset in LangSmith
    dataset = client.create_dataset(dataset_name="Text Summarizer Q&A Dataset")

    # Test cases
    examples = [
        # Case 1: Summary Tests
        {
            "inputs": {
                "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
                "question": "Summarize the main concepts of artificial intelligence in 3 sentences"
            },
            "outputs": {
                "expected_answer": "Should provide a concise 3-sentence summary covering AI definition, key applications, and current state"
            },
        },
        {
            "inputs": {
                "url": "https://realpython.com/python-introduction/",
                "question": "Summarize what Python programming language is in 3 sentences"
            },
            "outputs": {
                "expected_answer": "Should provide a concise 3-sentence summary covering Python's nature, features, and popularity from the blog perspective"
            },
        },
        
        # Case 2: Specific Context Questions
        {
            "inputs": {
                "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
                "question": "What are the main branches or types of AI mentioned?"
            },
            "outputs": {
                "expected_answer": "Should mention specific AI types like narrow AI, general AI, machine learning, etc. based on context"
            },
        },
        {
            "inputs": {
                "url": "https://realpython.com/python-introduction/",
                "question": "What are the key advantages of Python mentioned in this article?"
            },
            "outputs": {
                "expected_answer": "Should mention Python's advantages like readability, simplicity, extensive libraries, community support, etc. based on the blog content"
            },
        },
        
        # Case 3: Out of Context Questions
        {
            "inputs": {
                "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
                "question": "What is the current stock price of Tesla?"
            },
            "outputs": {
                "expected_answer": "Should respond that this information is not mentioned in the context or not available"
            },
        },
        {
            "inputs": {
                "url": "https://realpython.com/python-introduction/",
                "question": "What are the latest developments in blockchain technology?"
            },
            "outputs": {
                "expected_answer": "Should respond that this information is not mentioned in the context or not available"
            },
        },
    ]

    # Add examples to the dataset
    client.create_examples(dataset_id=dataset.id, examples=examples)

    print(f"Created dataset '{dataset.name}' with {len(examples)} examples")
    print(f"Dataset ID: {dataset.id}")
    return dataset

if __name__ == "__main__":
    # Create the dataset
    create_summarizer_dataset()