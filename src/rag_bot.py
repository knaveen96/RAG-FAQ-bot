import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from sentence_transformers import CrossEncoder
from config import OPENAI_API_KEY

USE_RERANKING = False   # Toggle reranking ON/OFF
TOP_K = 5              # Retrieve more docs if reranking is enabled
FINAL_K = 1            # Final number of docs to use

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)

vectorstore = FAISS.load_local(
    "data/index/founderstribune_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=OPENAI_API_KEY
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    output_key="answer",
    return_messages=True
)

system_template = """
You are a helpful assistant knowledgeable about tech founders.
Use the following context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Answer everything in English and give detailed answers.
----------------
{context}
"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("Chat History:\n{chat_history}\n\nQuestion:\n{question}"),
]

qa_prompt = ChatPromptTemplate.from_messages(messages)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,  # we will optionally rerank before passing
    memory=memory,
    return_source_documents=True,
    chain_type="stuff",
    combine_docs_chain_kwargs={'prompt': qa_prompt}
)

# Optional reranker model
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") if USE_RERANKING else None

def rerank_documents(query, docs):
    """Re-rank docs using cross-encoder."""
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked[:FINAL_K]]


def ask_faq(user_query: str) -> str:
    # Retrieve docs
    docs = retriever.get_relevant_documents(user_query)

    if USE_RERANKING and reranker:
        docs = rerank_documents(user_query, docs)

    # Run QA chain with either reranked or raw docs
    result = qa_chain({"question": user_query})

    answer = result["answer"]
    sources = [doc.metadata.get("url", "N/A") for doc in docs]

    formatted_sources = "\n".join(f"- {s}" for s in sources)
    return answer
