# RAG-Powered Q&A Summarizer

A web content processing and question-answering system built with LangChain and Groq API. Uses RAG (Retrieval-Augmented Generation) to answer questions about web content.

## Prerequisites

- Python 3.8 or higher
- Groq API key (free at [console.groq.com](https://console.groq.com))

## Setup

1. **Clone or download this project**

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```bash
   touch .env
   ```
   
   Add your Groq API key to `.env`:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Getting Your Groq API Key

1. Visit [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key to your `.env` file

## Project Structure

```
text-summarizer-app/
├── src/                    # Main application code
│   ├── summarizerBot.py   # Main functions: process_url() & ask_question()
│   ├── url_fetcher.py     # Web content fetching with BeautifulSoup
│   ├── rag_chain.py       # RAG workflow orchestration with LCEL
│   ├── retriever.py       # TF-IDF document retrieval & vector storage
│   └── README.md          # Detailed component documentation
├── evaluation/             # Evaluation system
│   └── dataset_setup.py   # LangSmith dataset & testing functions
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Running the System

### Quick Start
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run the example (processes URL + asks questions)
python src/summarizerBot.py
```

## How It Works

The system uses a **two-step RAG approach**:

### Step 1: URL Processing (`process_url()`)
1. **🌐 Content Extraction**: Fetches web pages using BeautifulSoup
2. **✂️ Content Chunking**: Splits text into 1000-character chunks with 200-character overlap
3. **🔢 Vector Storage**: Creates TF-IDF embeddings and stores in vector database
4. **⚙️ Chain Initialization**: Sets up LangChain LCEL retrieval chain

### Step 2: Question Answering (`ask_question()`)
1. **🔍 Context Retrieval**: Finds top 3 most relevant chunks using TF-IDF similarity
2. **📝 Context Formatting**: Combines retrieved chunks into context
3. **🤖 Response Generation**: Groq LLaMA model generates contextual answers
4. **💡 Structured Output**: Returns formatted answer based on retrieved content


## Evaluation & Testing

```bash
# Setup datasets
python evaluation/dataset_setup.py
```

