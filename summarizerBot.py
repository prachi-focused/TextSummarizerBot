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
    try:
        prompt = prompt_template.invoke(state["messages"])
        response = model.invoke(prompt)
        return {"messages": response}
    except openai.RateLimitError as e:
        print(f"Rate limit exceeded: {e}")
        print("Please check your OpenAI billing and quota at https://platform.openai.com/usage")
        # Return a fallback message
        from langchain_core.messages import AIMessage
        fallback_message = AIMessage(content="I'm sorry, but I've exceeded my API quota. Please check your OpenAI billing or try again later.")
        return {"messages": fallback_message}
    except Exception as e:
        print(f"An error occurred: {e}")
        from langchain_core.messages import AIMessage
        fallback_message = AIMessage(content="I'm sorry, but an error occurred while processing your request.")
        return {"messages": fallback_message}


# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Configuration for conversation history
CONVERSATION_WINDOW_SIZE = 10 

config = {"configurable": {"thread_id": "abc123"}}

query = "Hi! I'm Bob."

input_messages = [HumanMessage(query)]

try:
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()
except Exception as e:
    print(f"Application error: {e}")
    print("The summarizer bot encountered an issue. Please try again later.") 