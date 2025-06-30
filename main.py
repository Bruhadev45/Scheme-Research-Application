import streamlit as st
import re
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.schema import Document
from utils.helpers import (
    load_api_key, load_and_split_documents,
    create_or_load_faiss_index, query_llm
)
import openai

st.set_page_config(page_title="Scheme Research Tool", layout="wide")
st.title("📘 Government Scheme Research Tool")

def is_valid_url(s):
    return re.match(r'^https?://', s)

# Sidebar input mode
st.sidebar.header("Input Section")
input_mode = st.sidebar.radio("Choose input method:", ("URL", "Upload Scheme File"))

url_list = []
docs = []

if input_mode == "URL":
    urls_input = st.sidebar.text_area("Enter scheme article URLs (one per line):", height=150)
    if urls_input.strip():
        url_list = [url.strip() for url in urls_input.split("\n") if is_valid_url(url.strip())]

elif input_mode == "Upload Scheme File":
    uploaded_file = st.sidebar.file_uploader("Upload a .pdf or .txt file", type=["pdf", "txt"])
    if uploaded_file is not None:
        try:
            # Save temporarily
            file_ext = uploaded_file.name.split('.')[-1]
            file_path = f"temp_uploaded_file.{file_ext}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            loader = UnstructuredFileLoader(file_path)
            docs = loader.load()

            if not docs:
                st.sidebar.warning("⚠️ File loaded but no text found.")
            else:
                st.sidebar.success("✅ File loaded and parsed successfully.")

        except Exception as e:
            st.sidebar.error(f"❌ Failed to read or parse file: {e}")

# Process button
process_btn = st.sidebar.button("🔍 Process")

# Question input
query = st.text_input("Ask a question about the scheme:")
submit_btn = st.button("Get Answer")

# Session state for vectorstore
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Process input (URL or File)
if process_btn:
    try:
        api_key = load_api_key()
        openai.api_key = api_key
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        if input_mode == "URL":
            if not url_list:
                st.warning("⚠️ Please enter valid URLs.")
            else:
                with st.spinner("🔄 Loading and processing URLs..."):
                    docs = load_and_split_documents(url_list)

        if not docs:
            st.error("❌ No documents found to process.")
        else:
            with st.spinner("🔧 Creating FAISS index..."):
                vectorstore = create_or_load_faiss_index(docs, embeddings)
                st.session_state.vectorstore = vectorstore
                st.success("✅ Content processed and indexed successfully.")

    except Exception as e:
        st.error(f"🚨 Error during processing: {e}")

# Handle query
if submit_btn:
    if st.session_state.vectorstore is None:
        st.warning("⚠️ Please process input first.")
    elif not query.strip():
        st.warning("⚠️ Enter a question.")
    else:
        with st.spinner("🤖 Generating answer..."):
            try:
                response, sources = query_llm(st.session_state.vectorstore, query)
                st.subheader("🔎 Answer")
                st.markdown(response)
                st.markdown("**📎 Sources:**")
                for src in sources:
                    st.markdown(f"- {src}")
            except Exception as e:
                st.error(f"🚨 Failed to generate answer: {e}")
