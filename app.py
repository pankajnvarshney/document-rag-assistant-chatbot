import streamlit as st
from src.rag_pipeline import RAGPipeline

st.set_page_config(page_title="Legal Chatbot", layout="wide")
st.title("📜 AI-Powered Legal Document Chatbot")

rag = RAGPipeline("vectordb/index.faiss", "vectordb/chunks.pkl")

if "messages" not in st.session_state:
    st.session_state.messages = []

query = st.chat_input("Ask a question about the document...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.spinner("Searching..."):
        answer, sources = rag.ask(query)
        st.session_state.messages.append({"role": "bot", "content": answer})
        with st.expander("📚 Source Chunks"):
            for i, src in enumerate(sources):
                st.markdown(f"**Chunk {i+1}:** {src}")

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if st.button("🔄 Clear Chat"):
    st.session_state.messages = []

