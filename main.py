# main.py

import streamlit as st
from utils.helpers import (
    load_api_key, load_and_split_documents,
    create_or_load_faiss_index, query_llm,
    extract_text_from_file, clean_text
)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import openai
import os
import shutil
import logging

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)

# Fix for OpenMP errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Generate structured summary
def generate_summary(text, api_key):
    prompt = f"""
Summarize the scheme information into the following four key sections. If a section is not mentioned in the text, write "**Not mentioned**".

Format each section using the following headers:
- 🏆 Scheme Benefits
- 📝 Application Process
- ✅ Eligibility
- 📄 Documents Required

Use bullet points under each section.

Content:
{text}
"""
    try:
        llm = ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo", openai_api_key=api_key)
        return llm.predict(prompt)
    except Exception as e:
        logging.error(f"Summary generation failed: {e}")
        return f"Summary generation failed: {e}"

# Main app
def main():
    st.set_page_config(page_title="Scheme Research Tool", layout="centered")
    st.title("🧾 Automated Scheme Research Tool")

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "summary" not in st.session_state:
        st.session_state.summary = ""

    with st.sidebar:
        st.header("📂 Input")
        input_mode = st.radio("Choose Input Type", ["Enter URLs", "Upload Files"])

        if input_mode == "Enter URLs":
            urls_input = st.text_area("Paste one or more scheme URLs:", height=200)
        else:
            uploaded_files = st.file_uploader("Upload multiple PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

        process_btn = st.button("📥 Process")
        clear_btn = st.button("♻️ Clear All")

    query = st.text_input("💬 Ask a question:")
    submit_btn = st.button("🤖 Get Answer")

    if clear_btn:
        st.session_state.vectorstore = None
        st.session_state.processed = False
        st.session_state.summary = ""
        if os.path.exists("faiss_store_openai"):
            shutil.rmtree("faiss_store_openai")
            logging.info("Cleared FAISS index and session state.")
        st.rerun()

    if process_btn:
        try:
            api_key = load_api_key()
            if not api_key:
                st.error("❌ OpenAI API key not found.")
                return

            openai.api_key = api_key
            all_docs = []

            if os.path.exists("faiss_store_openai"):
                shutil.rmtree("faiss_store_openai")

            with st.spinner("🔄 Extracting and indexing documents..."):
                if input_mode == "Upload Files" and uploaded_files:
                    for file in uploaded_files:
                        raw_text = extract_text_from_file(file)
                        cleaned = clean_text(raw_text)
                        docs = load_and_split_documents(file_content=cleaned, file_name=file.name)
                        all_docs.extend(docs)
                        logging.info(f"Processed uploaded file: {file.name}")

                elif input_mode == "Enter URLs":
                    url_list = [u.strip() for u in urls_input.split("\n") if u.strip()]
                    docs = load_and_split_documents(url_list=url_list)
                    all_docs.extend(docs)
                    logging.info(f"Processed URLs: {url_list}")

                if all_docs:
                    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                    vs = create_or_load_faiss_index(all_docs, embeddings)
                    st.session_state.vectorstore = vs
                    st.session_state.processed = True
                    logging.info("FAISS index created successfully.")
                    st.success("✅ Documents processed successfully.")

                    with st.spinner("📝 Generating structured summary..."):
                        try:
                            combined_text = "\n\n".join([doc.page_content for doc in all_docs[:10]])
                            summary = generate_summary(combined_text, api_key)
                            st.session_state.summary = summary
                            logging.info("Structured summary generated successfully.")
                        except Exception as e:
                            st.session_state.summary = f"Summary generation failed: {e}"
                            logging.error(f"Summary generation after processing failed: {e}")
                else:
                    logging.warning("No valid documents found to process.")
                    st.error("❌ No valid content found to process.")

        except Exception as e:
            logging.error(f"Processing error: {e}")
            st.error(f"🚨 Processing error: {e}")

    if st.session_state.summary:
        st.subheader("🧾 Scheme Summary")
        st.markdown(st.session_state.summary)

    if submit_btn:
        try:
            api_key = load_api_key()
            if not api_key:
                st.error("❌ OpenAI API key not found.")
                return

            openai.api_key = api_key

            if not query.strip():
                st.warning("⚠️ Please enter a question.")
            elif not st.session_state.vectorstore:
                st.warning("⚠️ Process documents before asking questions.")
            else:
                with st.spinner("🤖 Generating answer..."):
                    answer, sources = query_llm(
                        st.session_state.vectorstore, query, model_name="gpt-3.5-turbo", k=5
                    )

                    st.subheader("🔎 Answer")
                    st.markdown(answer)

                    if sources:
                        st.subheader("📎 Sources")
                        for src in sorted(set(sources)):
                            st.markdown(f"- [{src}]({src})")

                    logging.info(f"Query answered: {query}")

        except Exception as e:
            logging.error(f"Answering error: {e}")
            st.error(f"🚨 Answering error: {e}")

if __name__ == "__main__":
    main()
