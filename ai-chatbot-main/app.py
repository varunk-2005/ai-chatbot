import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
import os

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from datetime import datetime
import asyncio

# ---- Load API Key ----
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", None)

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    st.session_state.vector_store = vector_store


def get_conversational_chain(api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, say "answer is not available in the context". Do not hallucinate.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """.strip()
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question, api_key, pdf_docs, conversation_history):
    if api_key is None or pdf_docs is None:
        st.warning("Please upload PDF files before asking a question.")
        return

    # Build vector store from PDFs
    raw_text = get_pdf_text(pdf_docs)
    chunks = get_text_chunks(raw_text)
    get_vector_store(chunks)

    # Retrieve relevant chunks from session state
    docs = st.session_state.vector_store.similarity_search(user_question)

    # Get answer from Gemini
    chain = get_conversational_chain(api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    user_question_output = user_question
    response_output = response['output_text']

    pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []
    conversation_history.append(
        (user_question_output, response_output, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ", ".join(pdf_names))
    )

    # ---- UI styles ----
    st.markdown(
        """
        <style>
            .section-title {
                font-weight: 600;
                font-size: 1rem;
                opacity: 0.85;
                margin: 0.25rem 0 0.5rem 0;
            }
            .chat-wrap { max-width: 900px; margin: 0 auto; }
            .chat-message {
                padding: 1rem 1.25rem;
                border-radius: 14px;
                margin: 0.5rem 0 1rem 0;
                display: flex;
                gap: 12px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.08);
                border: 1px solid rgba(255,255,255,0.06);
            }
            .chat-message.user { background: linear-gradient(180deg, #2f3542 0%, #2a2f3a 100%); }
            .chat-message.bot  { background: linear-gradient(180deg, #3c4456 0%, #353d4d 100%); }
            .chat-message .avatar { flex: 0 0 52px; }
            .chat-message .avatar img {
                width: 52px; height: 52px; border-radius: 50%; object-fit: cover; border: 1px solid rgba(255,255,255,0.1);
            }
            .chat-message .message { flex: 1; color: #f7f7f7; line-height: 1.55; }
            .chat-message .meta { font-size: 0.8rem; opacity: 0.75; margin-top: 0.35rem; }
            hr.clean { border: none; border-top: 1px solid rgba(255,255,255,0.1); margin: 0.75rem 0 1rem; }
            .sidebar-note { font-size: 0.9rem; opacity: 0.8; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ---- Render current Q/A ----
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Latest exchange</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="chat-message user">
            <div class="avatar">
                <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
            </div>
            <div class="message">{user_question_output}</div>
        </div>
        <div class="chat-message bot">
            <div class="avatar">
                <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp">
            </div>
            <div class="message">{response_output}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---- Conversation history ----
    history_to_show = conversation_history[:-1]

    if len(history_to_show) > 0:
        st.markdown('<hr class="clean" />', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Previous messages</div>', unsafe_allow_html=True)

    for question, answer, timestamp, pdf_name in reversed(history_to_show):
        st.markdown(
            f"""
            <div class="chat-message user">
                <div class="avatar">
                    <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
                </div>
                <div class="message">{question}
                    <div class="meta">Asked: {timestamp} · PDFs: {pdf_name}</div>
                </div>
            </div>
            <div class="chat-message bot">
                <div class="avatar">
                    <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp">
                </div>
                <div class="message">{answer}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # ---- CSV download in sidebar ----
    if len(st.session_state.conversation_history) > 0:
        df = pd.DataFrame(
            st.session_state.conversation_history,
            columns=["Question", "Answer", "Timestamp", "PDF Name"]
        )
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a class="sidebar-note" href="data:file/csv;base64,{b64}" download="conversation_history.csv">Download conversation history (CSV)</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
        st.sidebar.caption("Exports all questions/answers with timestamps and source PDF names.")


def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.title("Chat with multiple PDFs")
    st.caption("Upload PDFs, then ask questions powered by local embeddings + Gemini.")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None

    api_key = GOOGLE_API_KEY
    if not api_key:
        st.sidebar.error("GOOGLE_API_KEY missing. Add it to your .env or Streamlit secrets.")
        return

    with st.sidebar:
        st.subheader("Controls")
        col1, col2 = st.columns(2)
        reset_button = col2.button("Reset")
        clear_button = col1.button("Rerun")

        if reset_button:
            st.session_state.conversation_history = []
            st.session_state.vector_store = None
            st.session_state.user_question = None

        if clear_button:
            if 'user_question' in st.session_state:
                st.warning("The previous query will be discarded.")
                st.session_state.user_question = ""
                if len(st.session_state.conversation_history) > 0:
                    st.session_state.conversation_history.pop()
            else:
                st.warning("The question in the input will be queried again.")

        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True,
            type=["pdf"],
            help="You can select multiple PDFs at once."
        )
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    pass
                st.success("Ready. Ask your question below.")
            else:
                st.warning("Please upload PDF files before processing.")

    user_question = st.text_input("Your question about the uploaded PDFs")
    if user_question:
        user_input(user_question, api_key, pdf_docs, st.session_state.conversation_history)
        st.session_state.user_question = ""


if __name__ == "__main__":
    main()