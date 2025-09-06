import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from config import GEMINI_API_KEY

#Load the FAISS index built in ingest.py
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY
)
vectorstore = FAISS.load_local("data/index/faq_index", embeddings, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

def ask_faq(question: str) -> str:
    # Takes a user question and returns an answer using RAG
    result = qa_chain({"query": question})
    answer = result["result"]

    # Optional: include source context for transparency
    sources = [doc.metadata.get("source", "N/A") for doc in result["source_documents"]]
    return answer
