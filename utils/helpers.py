# Import necessary modules
import configparser
import os
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.schema import Document

import streamlit as st

openai_key = st.secrets["api_keys"]["openai_key"]

# Load OpenAI API key from a .config file
def load_api_key():
    config = configparser.ConfigParser()
    config.read('.config')
    return config['OPENAI']['api_key']

# Clean up raw text by removing excessive spaces, newlines, and non-breaking spaces
def clean_text(text):
    return " ".join(text.replace("\xa0", " ").replace("\n", " ").split())

# Extract text content from an uploaded file (TXT or PDF)
def extract_text_from_file(file):
    ext = file.name.split(".")[-1].lower()
    if ext == "txt":
        return file.read().decode("utf-8")
    elif ext == "pdf":
        reader = PdfReader(file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    return ""

# Load and split documents from either a list of URLs or uploaded file content
def load_and_split_documents(url_list=None, file_content=None, file_name=None):
    try:
        docs = []

        # If URLs are provided, process them
        if url_list:
            for url in url_list:
                if url.lower().endswith(".pdf"):
                    # Download and extract text from PDF
                    response = requests.get(url)
                    if response.ok:
                        with open("temp.pdf", "wb") as f:
                            f.write(response.content)
                        reader = PdfReader("temp.pdf")
                        text = "\n".join([page.extract_text() or "" for page in reader.pages])
                        docs.append(Document(page_content=text, metadata={"source": url}))
                else:
                    # Use UnstructuredURLLoader for non-PDF URLs (HTML pages etc.)
                    loader = UnstructuredURLLoader(urls=[url])
                    docs.extend(loader.load())

        # If file content is provided (from upload), create a Document object
        elif file_content:
            docs = [Document(page_content=file_content, metadata={"source": file_name or "uploaded_file"})]

        # If documents exist, split them into smaller chunks for better embedding and retrieval
        if docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            return splitter.split_documents(docs)

        return []
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []

# Create or load a FAISS vector index from documents
def create_or_load_faiss_index(docs, embeddings, index_path="faiss_store_openai"):
    try:
        # If index already exists locally, load it
        if os.path.exists(index_path):
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

        # Otherwise, create a new FAISS index and save it
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(index_path)
        return vectorstore
    except Exception as e:
        print(f"FAISS error: {e}")
        return None

# Query the LLM using the vectorstore and return both the answer and the sources used
def query_llm(vectorstore, query, model_name="gpt-3.5-turbo", k=3):
    try:
        # Retrieve top-k relevant chunks using vector similarity
        relevant_docs = vectorstore.similarity_search(query, k=k)
        if not relevant_docs:
            return "No relevant content found.", []

        # Extract source links and combine content for context
        sources = [doc.metadata.get("source", "N/A") for doc in relevant_docs]
        combined = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Load API key and set up ChatOpenAI LLM
        api_key = load_api_key()
        llm = ChatOpenAI(temperature=0.4, model_name=model_name, openai_api_key=api_key)

        # Construct prompt using retrieved document content
        prompt = f"""
Use the following document content to answer the question clearly and in detail.

---
{combined}
---

User's question: {query}

If the answer is not found in the content, say: "The document does not provide that information."
"""
        return llm.predict(prompt), sources
    except Exception as e:
        return f"Failed to generate answer: {e}", []
