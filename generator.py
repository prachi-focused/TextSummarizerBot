# Simple text generator
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

class TextGenerator:
    def __init__(self):
        # Load environment and setup model
        load_dotenv()
        self.model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
        
        # Simple prompt template
        self.prompt = ChatPromptTemplate.from_template("""
            Context: {context}
            Question: {question}
            
            Provide a clear, concise summary of the content:
        """)
        print("Generator ready")
    
    def generate_response(self, context: str, question: str) -> str:
        """Generate response from context and question"""
        try:
            chain = self.prompt | self.model | StrOutputParser()
            return chain.invoke({"context": context, "question": question})
        except Exception as e:
            return f"Error generating response: {e}"