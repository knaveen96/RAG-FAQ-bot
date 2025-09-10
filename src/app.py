import streamlit as st
from rag_bot import ask_faq

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="📚 FAQ RAG Bot", page_icon="🤖")
st.title("📚 FAQ Chatbot with RAG & Memory")

# -----------------------------
# Session state to store conversation
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# User input
# -----------------------------
user_input = st.chat_input("Ask me a question...")

if user_input:
    response = ask_faq(user_input)

    # Store messages in session
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

# -----------------------------
# Display chat history
# -----------------------------
for role, message in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**🧑 {role}:** {message}")
    else:
        st.markdown(f"**🤖 {role}:** {message}")
