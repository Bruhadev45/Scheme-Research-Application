# utils/helpers.py

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

# Load OpenAI API key from secrets or fallback to .config file
def load_api_key():
    try:
        return st.secrets["api_keys"]["openai_key"]
    except Exception:
        config = configparser.ConfigParser()
        config.read('.config')
        try:
            return config['OPENAI']['api_key']
        except Exception:
            return None

def clean_text(text):
    return " ".join(text.replace("\xa0", " ").replace("\n", " ").split())

def extract_text_from_file(file):
    ext = file.name.split(".")[-1].lower()
    if ext == "txt":
        return file.read().decode("utf-8")
    elif ext == "pdf":
        reader = PdfReader(file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    return ""

def load_and_split_documents(url_list=None, file_content=None, file_name=None):
    try:
        docs = []

        if url_list:
            for url in url_list:
                if url.lower().endswith(".pdf"):
                    response = requests.get(url)
                    if response.ok:
                        with open("temp.pdf", "wb") as f:
                            f.write(response.content)
                        reader = PdfReader("temp.pdf")
                        text = "\n".join([page.extract_text() or "" for page in reader.pages])
                        docs.append(Document(page_content=text, metadata={"source": url}))
                else:
                    loader = UnstructuredURLLoader(urls=[url])
                    docs.extend(loader.load())

        elif file_content:
            docs = [Document(page_content=file_content, metadata={"source": file_name or "uploaded_file"})]

        if docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            return splitter.split_documents(docs)

        return []
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []

def create_or_load_faiss_index(docs, embeddings, index_path="faiss_store_openai"):
    try:
        if os.path.exists(index_path):
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(index_path)
        return vectorstore
    except Exception as e:
        print(f"FAISS error: {e}")
        return None

def query_llm(vectorstore, query, model_name="gpt-3.5-turbo", k=3):
    try:
        relevant_docs = vectorstore.similarity_search(query, k=k)
        if not relevant_docs:
            return "No relevant content found.", []

        sources = [doc.metadata.get("source", "N/A") for doc in relevant_docs]
        combined = "\n\n".join([doc.page_content for doc in relevant_docs])

        api_key = load_api_key()
        llm = ChatOpenAI(temperature=0.4, model_name=model_name, openai_api_key=api_key)

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
