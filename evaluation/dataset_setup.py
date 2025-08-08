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
                "url": "https://bigthink.com/starts-with-a-bang/nuclear-reactor-moon-nasa/?utm_source=firefox-newtab-en-us",
                "question": "Summarize the blog post in 3 sentences"
            },
            "outputs": {
                "answer_criteria": "Should provide a concise 3-sentence summary covering the Moon’s power challenges, NASA’s nuclear reactor plan, and its broader implications."
            },
        },
        {
            "inputs": {
                "url": "https://realpython.com/python-introduction/",
                "question": "Summarize what Python programming language is in 3 sentences"
            },
            "outputs": {
                "answer_criteria": "Should provide a concise 3-sentence summary covering Python's purpose, features, and use cases mentioned in the blog post"
            },
        },
        
        # Case 2: Specific Context Questions
        {
            "inputs": {
                "url": "https://bigthink.com/starts-with-a-bang/nuclear-reactor-moon-nasa/?utm_source=firefox-newtab-en-us",
                "question": "Is there a nuclear reactor planned for the Moon?"
            },
            "outputs": {
                "answer_criteria": "Should mention the specific power output (in kilowatts) and the targeted deployment timeline."
            },
        },
        {
            "inputs": {
                "url": "https://realpython.com/python-introduction/",
                "question": "What are the key advantages of Python mentioned in this article?"
            },
            "outputs": {
                "answer_criteria": "Should mention Python's advantages as a programming language."
            },
        },
        
        # Case 3: Out of Context Questions
        {
            "inputs": {
                "url": "https://bigthink.com/starts-with-a-bang/nuclear-reactor-moon-nasa/?utm_source=firefox-newtab-en-us",
                "question": "What is the distance between the Earth and the Moon?"
            },
            "outputs": {
                "answer_criteria": "Should respond that this information is not mentioned in the context."
            },
        },
        {
            "inputs": {
                "url": "https://realpython.com/python-introduction/",
                "question": "What are the latest developments in blockchain technology?"
            },
            "outputs": {
                "answer_criteria": "Should respond that this information is not mentioned in the context."
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