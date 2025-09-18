import os
import json
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import OPENAI_API_KEY

RAW_DIR = "data/raw/founderstribune"
INDEX_DIR = "data/index/founderstribune_index"

def load_json_docs():
    docs = []
    # Walk through subfolders by date
    for root, _, files in os.walk(RAW_DIR):
        for filename in files:
            if filename.endswith(".json"):
                path = os.path.join(root, filename)
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Build LangChain Document
                text = data.get("text", "").strip()
                if not text:
                    continue

                metadata = {
                    "title": data.get("title", ""),
                    "author": data.get("author", ""),
                    "date": data.get("date", ""),
                    "url": data.get("url", ""),
                }

                docs.append(Document(page_content=text, metadata=metadata))
    return docs

def build_faiss_index():
    docs = load_json_docs()
    if not docs:
        raise RuntimeError(f"❌ No JSON documents found in {RAW_DIR}/")

    print(f"✅ Loaded {len(docs)} articles")

    # Split documents into smaller chunks for embeddings
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"✅ Split into {len(chunks)} chunks")

    # Create embeddings using OpenAI cheapest embedding model
    print("⏳ Creating embeddings with OpenAI...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # Cheapest embedding model
        openai_api_key=OPENAI_API_KEY,
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save FAISS index locally
    os.makedirs(os.path.dirname(INDEX_DIR), exist_ok=True)
    vectorstore.save_local(INDEX_DIR)
    print(f"✅ FAISS index saved at {INDEX_DIR}")

if __name__ == "__main__":
    build_faiss_index()
