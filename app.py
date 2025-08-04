import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings  # Corrected import for OpenAIEmbeddings
from langchain.vectorstores import FAISS  # Corrected import for FAISS vector store
from langchain.chat_models import ChatOpenAI  # Corrected import for ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit app configuration
st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 PDF Question Answering Chatbot")

# Initialize session state to store vector store
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# File upload section
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    # Initialize the vector store to None to avoid stale data
    st.session_state.vectorstore = None

    # Extract text from PDF
    try:
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}")
        st.stop()

    # Split the text into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,  # Adjust size to fit your content
        chunk_overlap=200,  # Small overlap between chunks
        length_function=len,
    )
    chunks = text_splitter.split_text(text)

    # Create embeddings and store them in FAISS vector store
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    st.session_state.vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    st.success("✅ PDF processed and embeddings created successfully!")

# QA interface for asking questions about the PDF
if st.session_state.vectorstore:
    st.header("💬 Ask Questions about the PDF")
    user_question = st.text_input("Your question:")

    if user_question:
        try:
            # Set up the retrieval-based QA chain
            qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(openai_api_key=openai_api_key, temperature=0),
                chain_type="stuff",
                retriever=st.session_state.vectorstore.as_retriever(),
                return_source_documents=False  # You can set to True to debug source chunks
            )

            result = qa.run(user_question)
            st.write("🔎 **Answer:**", result)

        except Exception as e:
            st.error(f"❌ Error generating response: {e}")
else:
    st.info("👆 Upload a PDF to begin.")

