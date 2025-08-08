import os
import sys
from dotenv import load_dotenv
from langsmith import Client
from langchain_groq import ChatGroq
import json

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
    answer_criteria = reference_outputs.get("answer_criteria", '')
    
    # Initialize LLM for evaluation
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    # Create evaluation prompt that returns JSON
    evaluation_prompt = f"""
        You are an expert evaluator. Your task is to determine if the answer is relevant to the question based on:
        - The content retrieved from the given URL
        - The expected answer criteria (which takes priority over content grounding)

        IMPORTANT: If the answer_criteria indicate that the system should respond with "not mentioned", "not available", "I don't know", or similar acknowledgments of missing information, then SKIP evaluating content grounding and focus only on whether the system correctly identifies the lack of information.

        Evaluate the following:
        1. Does the answer address the question being asked?
        2. Does the answer meet the expected answer_criteria?
        3. Is the answer grounded in the retrieved content? (Skip if the expected response is "no information found")

        Question: {question}
        Answer: {answer}
        Retrieved Content: {retrieved_context}
        Answer Criteria: {answer_criteria}

        Scoring Guidelines:
        - 8–10: Fully meets criteria, including appropriate "no info" responses
        - 6–7: Mostly meets criteria with minor issues
        - 4–5: Partially meets criteria with notable gaps
        - 2–3: Poor match to criteria or unclear response
        - 1: Completely wrong or hallucinates when info is expected to be missing

        Return ONLY a valid JSON object in this exact format (no extra commentary):
        {{
            "score": <number from 1 to 10>,
            "explanation": "<Brief explanation (2-3 sentences) of your reasoning>"
        }}
    """
    
    try:
        # Get evaluation from LLM
        response = llm.invoke(evaluation_prompt)
        evaluation_text = response.content.strip()
        
        # Parse JSON response
        evaluation_data = json.loads(evaluation_text)
        
        score = float(evaluation_data.get("score", 5.0))
        score = min(max(score, 0), 10)  # Clamp between 0-10
        explanation = evaluation_data.get("explanation", "No explanation provided")
        
        return {
            "key": "relevance_score",
            "score": score,
            "value": f"{score}/10",
            "comment": explanation
        }
        
    except json.JSONDecodeError as e:
        return {
            "key": "relevance_score",
            "score": 0.0,
            "value": "0/10",
            "comment": f"Failed to parse JSON response: {evaluation_text[:100]}..."
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
        # target for evaluation
        target_function,
        # dataset to use
        data=DATASET_NAME,
        # evaluators to use
        evaluators=[relevance_evaluator],
        # name of the experiment
        experiment_prefix="relevance-eval",
        # number of concurrent runs
        max_concurrency=1, 
    )
    
    print("Evaluation completed successfully!")
    
except Exception as e:
    print(f"Evaluation failed: {e}")
    print("Make sure you have LANGCHAIN_API_KEY set in your .env file")
