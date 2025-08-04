"""
Dataset Setup for Text Summarizer Evaluation
This file creates a LangSmith dataset using the new summarizer functions for evaluating RAG-based Q&A.
"""

from langsmith import Client
import sys
import os

# Add src directory to path to import summarizer functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from summarizerBot import process_url, ask_question

def create_summarizer_dataset():
    """Create a dataset with URL + question pairs for summarizer evaluation."""
    
    client = Client()

    # Create dataset for text summarizer evaluation
    dataset = client.create_dataset(
        dataset_name="RAG Summarizer Q&A Dataset", 
        description="A dataset of URLs and questions for evaluating RAG-based question answering and summarization."
    )

    # Sample URL + question pairs for testing different types of content and queries
    examples = [
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
                "url": "https://en.wikipedia.org/wiki/Machine_learning",
                "question": "What are the main types of machine learning?"
            },
            "outputs": {
                "expected_answer": "Should mention supervised, unsupervised, and reinforcement learning"
            },
        },
        {
            "inputs": {
                "url": "https://en.wikipedia.org/wiki/Natural_language_processing",
                "question": "What are the practical applications of NLP?"
            },
            "outputs": {
                "expected_answer": "Should list applications like chatbots, translation, sentiment analysis, etc."
            },
        },
        {
            "inputs": {
                "url": "https://en.wikipedia.org/wiki/Deep_learning",
                "question": "How do neural networks work in deep learning?"
            },
            "outputs": {
                "expected_answer": "Should explain neural network structure, layers, and learning process"
            },
        },
        {
            "inputs": {
                "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
                "question": "What makes Python popular for data science and AI?"
            },
            "outputs": {
                "expected_answer": "Should mention simplicity, libraries like NumPy/Pandas/TensorFlow, and ecosystem"
            },
        },
        {
            "inputs": {
                "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
                "question": "What are the ethical concerns around AI?"
            },
            "outputs": {
                "expected_answer": "Should discuss bias, job displacement, privacy, and control concerns"
            },
        },
    ]

    # Add examples to the dataset
    for example in examples:
        client.create_example(
            inputs=example["inputs"],
            outputs=example["outputs"],
            dataset_id=dataset.id,
        )

    print(f"Created dataset '{dataset.name}' with {len(examples)} examples")
    print(f"Dataset ID: {dataset.id}")
    return dataset

def test_summarizer_with_example():
    """Test the summarizer functions with a sample URL and question."""
    
    # Sample test case
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    question = "Summarize the main concepts of artificial intelligence in 3 sentences"
    
    print("=" * 60)
    print("Testing Summarizer Functions")
    print("=" * 60)
    
    # Step 1: Process URL
    print(f"1. Processing URL: {url}")
    status = process_url(url)
    print(f"   Status: {status}")
    
    if "initialized" in status.lower():
        print("\n2. Asking question...")
        # Step 2: Ask question
        answer = ask_question(question)
        
        print(f"\n3. Evaluation complete!")
        print(f"   Question: {question}")
        print(f"   Answer: {answer[:200]}..." if len(answer) > 200 else f"   Answer: {answer}")
    else:
        print(f"   Failed to process URL: {status}")

def run_dataset_evaluation_example():
    """Example of how to evaluate using the dataset structure."""
    
    # This shows how an evaluator would use the dataset
    example_input = {
        "url": "https://en.wikipedia.org/wiki/Machine_learning",
        "question": "What are the main types of machine learning?"
    }
    
    expected_output = {
        "expected_answer": "Should mention supervised, unsupervised, and reinforcement learning"
    }
    
    print("\n" + "=" * 60)
    print("Dataset Evaluation Example")
    print("=" * 60)
    print(f"Input URL: {example_input['url']}")
    print(f"Input Question: {example_input['question']}")
    print(f"Expected: {expected_output['expected_answer']}")
    
    # Process URL and ask question
    url_status = process_url(example_input['url'])
    
    if "initialized" in url_status.lower():
        actual_answer = ask_question(example_input['question'])
        
        print("\n" + "-" * 40)
        print("EVALUATION RESULT:")
        print(f"Actual Answer: {actual_answer}")
        print("-" * 40)
        
        return {
            "input": example_input,
            "expected": expected_output,
            "actual": actual_answer,
            "status": "success"
        }
    else:
        return {
            "input": example_input,
            "expected": expected_output,
            "actual": f"Failed to process URL: {url_status}",
            "status": "failed"
        }

if __name__ == "__main__":
    # Create the dataset
    create_summarizer_dataset()
    
    # Test the summarizer functions
    test_summarizer_with_example()
    
    # Show evaluation example
    run_dataset_evaluation_example()