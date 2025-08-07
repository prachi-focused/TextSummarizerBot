import os
import sys
from dotenv import load_dotenv
from langsmith import Client
from langchain_groq import ChatGroq

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from documentQABot import process_url
from rag_chain import RAGChain

# Load environment variables (for LangSmith API key)
load_dotenv()

# Connect to LangSmith
client = Client()

# Your dataset name from the creation step
DATASET_NAME = "Text Summarizer Q&A Dataset"

def target_function(inputs: dict) -> dict:
    """Target function that runs your RAG system and returns answer with context."""
    url = inputs.get("url")
    question = inputs.get("question")
    
    if not url or not question:
        return {"answer": "Error: Missing URL or question", "context": ""}
    
    # Initialize RAG chain
    rag_chain = RAGChain()
    
    # Process the URL first
    url_status = rag_chain.process_url(url)
    
    if "initialized" not in url_status.lower():
        return {"answer": f"Error processing URL: {url_status}", "context": ""}
    
    # Get answer and context
    result = rag_chain.query_data_with_context(question)
    
    return {
        "answer": result["answer"],
        "context": result["context"]
    }

def relevance_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    """LLM-based evaluator that checks if the answer is relevant to the question based on retrieved content."""
    question = inputs.get('question', '')
    answer = outputs.get('answer', '')
    retrieved_context = outputs.get('context', '')
    
    # Initialize LLM for evaluation
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    # Create evaluation prompt
    evaluation_prompt = f"""
        You are an expert evaluator. Your task is to determine if the answer is relevant to the question based on the content retrieved from the provided URL.

        Question: {question}
        Answer: {answer}
        Relevant Content: {retrieved_context}

        Evaluate whether the answer is relevant to the question based on the retrieved content. Consider:
        1. Does the answer address the question asked?
        2. Is the answer grounded in the retrieved content?
        3. Is the information in the answer factually consistent with the content?

        Provide your evaluation in this exact format:
        RELEVANCE_SCORE: [score from 1-10]
        EXPLANATION: [2-3 sentences explaining your evaluation]
    """
    
    try:
        # Get evaluation from LLM
        response = llm.invoke(evaluation_prompt)
        evaluation_text = response.content
        
        # Parse the response to extract score
        score = 5.0  # Default score
        explanation = "Could not parse evaluation"
        
        lines = evaluation_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if "RELEVANCE_SCORE:" in line:
                import re
                match = re.search(r'\d+(?:\.\d+)?', line.split("RELEVANCE_SCORE:")[-1])
                if match:
                    score = float(match.group())
                    score = min(max(score, 0), 10)  # Clamp between 0-10
            elif "EXPLANATION:" in line:
                explanation = line.split("EXPLANATION:")[-1].strip()
        
        return {
            "key": "relevance_score",
            "score": score,
            "value": f"{score}/10",
            "comment": explanation
        }
        
    except Exception as e:
        return {
            "key": "relevance_score",
            "score": 0.0,
            "value": "0/10",
            "comment": f"Evaluation failed: {str(e)}"
        }

# Run evaluation
try:
    results = client.evaluate(
        target_function,
        data=DATASET_NAME,
        evaluators=[relevance_evaluator],
        experiment_prefix="relevance-eval",
        max_concurrency=1,
    )
    
    print("Evaluation completed successfully!")
    
except Exception as e:
    print(f"Evaluation failed: {e}")
    print("Make sure you have LANGCHAIN_API_KEY set in your .env file")
