This project is a Retrieval-Augmented Generation (RAG) chatbot built using LangChain, OpenAI, and Streamlit.
It retrieves answers from custom documents (like scraped articles or FAQs), enhances them with LLM responses, and provides conversational memory.

🚀 Built with:
Python
LangChain (Retrieval + Chains)
FAISS (Vector database)
OpenAI Embeddings & LLMs
Streamlit (UI)
Custom Ranking Layer (optional, to improve retrieval relevance)

⚙️ Features
Scrapes and ingests documents into FAISS vector database
Uses text-embedding-3-small for embeddings (cheap & efficient)
Conversational memory with LangChain’s ConversationBufferMemory
Ranking toggle: compare retrieval with vs. without re-ranking
Streamlit UI for local testing

🗂️ Project Structure
RAG-FAQ-bot/
│
├── data/                    # Raw and indexed data
│   ├── raw/                 # Scraped JSON articles
│   └── index/               # FAISS vector index
│
├── src/
│   ├── app.py               # Streamlit UI
│   ├── rag.py               # RAG pipeline + ranking
│   ├── ingest.py            # Embedding + FAISS index builder
│   ├── scraper.py           # Scraper for new articles
│   └── config.py            # API key loading
│
├── requirements.txt         # Dependencies
└── README.md                # This file

🔑 Setup

Clone the repo
git clone <>
cd RAG-FAQ-bot


Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows

Install dependencies
pip install -r requirements.txt

Set your OpenAI API key
Create a .env file in the root folder:

OPENAI_API_KEY=your_openai_api_key_here

🛠️ Usage
1. Scrape data
python src/scraper.py

2. Build embeddings + FAISS index
python src/ingest.py

3. Run the chatbot
streamlit run src/app.py

Then open 👉 http://localhost:8501 in your browser.

🎯 Example Queries
"What do you know about Sam Altman?"

💡 Key Learnings
Document chunking with RecursiveCharacterTextSplitter improves retrieval
text-embedding-3-small is efficient for embeddings
Ranking (via reranker models) noticeably boosts answer precision
Adding conversational memory makes interactions feel more natural