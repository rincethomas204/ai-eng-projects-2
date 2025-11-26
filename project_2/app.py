import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

SYSTEM_TEMPLATE = """
You are a Customer Support Chatbot. Use only the information in CONTEXT to answer.
If the answer is not in CONTEXT, say: "I don't know based on the retrieved documents."
Be concise and accurate. Cite sources when possible as [source: SOURCE].

CONTEXT:
{context}

USER:
{question}
"""

@st.cache_resource
def build_chain():
    embedding_model = SentenceTransformerEmbeddings(model_name="thenlper/gte-small")
    vectordb = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 8})
    llm = Ollama(model="gemma3:1b", temperature=0.1)
    prompt = PromptTemplate(input_variables=["context", "question"], template=SYSTEM_TEMPLATE)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

st.set_page_config(page_title="RAG Chatbot Demo", page_icon="📚")
st.title("📚 RAG Chatbot Demo")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (question, answer)

chain = build_chain()

question = st.text_input("Ask a question about our products or policies:")

if question:
    with st.spinner("Thinking..."):
        result = chain.invoke({"question": question, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.append((question, result.get("answer", "")))

    st.write("**Answer:**", result.get("answer", ""))

    with st.expander("Show Source Documents"):
        for doc in result.get("source_documents", []):
            st.write(f"- {doc.metadata.get('source', 'Unknown source')}")
            st.write(doc.page_content)

if st.session_state.chat_history:
    st.divider()
    st.subheader("Conversation")
    for q, a in st.session_state.chat_history:
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")