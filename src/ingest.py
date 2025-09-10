import os
from langchain_community.document_loaders import TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import OPENAI_API_KEY

def build_faiss_index():
    docs = []

    # Load all text files from raw data folder
    raw_path = "data/raw"
    for filename in os.listdir(raw_path):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(raw_path, filename))
            docs.extend(loader.load())  # Add loaded docs to list

    if not docs:
        raise RuntimeError("❌ No .txt files found in data/raw/")

    print(f"✅ Loaded {len(docs)} documents")

    # Split documents into smaller chunks for embeddings
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"✅ Split into {len(chunks)} chunks")

    # Create embeddings using OpenAI cheapest embedding model
    print("⏳ Creating embeddings with OpenAI...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # Cheapest embedding model
        openai_api_key=OPENAI_API_KEY
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save FAISS index locally
    vectorstore.save_local("data/index/faq_index")
    print("✅ FAISS index saved at data/index/faq_index")

if __name__ == "__main__":
    build_faiss_index()
