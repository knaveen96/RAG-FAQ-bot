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
from config import OPENAI_API_KEY

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)

vectorstore = FAISS.load_local(
    "data/index/faq_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

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
You are a helpful assistant.
You MUST respond in Telugu.
Use the following context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
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
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    chain_type="stuff",
    combine_docs_chain_kwargs={'prompt': qa_prompt}
)

def ask_faq(user_query: str) -> str:
    result = qa_chain({"question": user_query})
    answer = result["answer"]
    sources = [doc.metadata.get("source", "N/A") for doc in result["source_documents"]]

    formatted_sources = "\n".join(f"- {s}" for s in sources)
    return answer
