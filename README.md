# Text Summarizer Bot

A simple text summarization bot built with LangChain and Groq API.

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

## Running the Bot

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run the bot
python summarizerBot.py
```

## How It Works

- The bot uses Groq's `llama-3.1-8b-instant` model
- It's configured to summarize text with conversation memory
- Current setup runs a simple test with "Hi! I'm Bob."

## Customizing

To summarize your own text, modify the `query` variable in `summarizerBot.py`:

```python
query = "Your text to summarize goes here..."
```

## Files

- `summarizerBot.py` - Main application
- `requirements.txt` - Python dependencies
- `.env` - Environment variables (create this file)
- `.gitignore` - Protects sensitive files

## Features

- ✅ Fast summarization with Groq API
- ✅ Conversation memory with LangGraph
- ✅ Error handling
- ✅ Free API tier available