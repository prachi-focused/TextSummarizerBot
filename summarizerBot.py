# Import relevant functionality
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, MessagesState
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import openai
import os

# Load environment variables from .env file
load_dotenv()

# Define the model
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=4096,
    timeout=60,
    max_retries=2
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that summarizes text.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define a new graph
workflow = StateGraph(state_schema=MessagesState)


# Define the function that calls the model
def call_model(state: MessagesState):
    prompt = prompt_template.invoke(state["messages"])
    response = model.invoke(prompt)
    return {"messages": response}
    

# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Configuration for conversation history
CONVERSATION_WINDOW_SIZE = 10 

config = {"configurable": {"thread_id": "abc123"}}

def read_text_file(file_path):
    """Read text content from a file."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found.")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            
        if not content:
            raise ValueError(f"File '{file_path}' is empty.")
            
        return content
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def summarize_text_from_file(file_path):
    """Read text from file and create a summarization query."""
    # Read the text content from file
    text_content = read_text_file(file_path)
    
    if text_content is None:
        return
    
    # Create query with the file content
    query = f"Please provide a concise summary of the following text:\n\n{text_content}"
    
    input_messages = [HumanMessage(query)]
    
    try:
        print(f"Summarizing content from: {file_path}")
        print("=" * 50)
        output = app.invoke({"messages": input_messages}, config)
        output["messages"][-1].pretty_print()
    except Exception as e:
        print(f"Application error: {e}")
        print("The summarizer bot encountered an issue. Please try again later.")

# Specify the text file to summarize
TEXT_FILE_PATH = "sample_text.txt"

# Run the summarization
summarize_text_from_file(TEXT_FILE_PATH) 