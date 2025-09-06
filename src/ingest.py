import os
from langchain_community.document_loaders import TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_google_genai import GoogleGenerativeAIEmbeddings     
from langchain_community.vectorstores import FAISS             
from config import GEMINI_API_KEY 

def build_faiss_index():
    """Load docs, split, embed, and save FAISS index."""

    docs = []
    raw_path = "data/raw"
    for filename in os.listdir(raw_path):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(raw_path, filename))
            docs.extend(loader.load())  # add to docs list

    if not docs:
        raise RuntimeError("❌ No .txt files found in data/raw/")

    print(f"✅ Loaded {len(docs)} documents")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"✅ Split into {len(chunks)} chunks")

    print("⏳ Creating embeddings with Gemini...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY 
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local("data/index/faq_index")
    print("✅ FAISS index saved at data/index/faq_index")

if __name__ == "__main__":
    build_faiss_index()
