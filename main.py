# Import necessary libraries and helper functions
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

# Fix for potential OpenMP runtime error on some machines
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Function to generate a brief summary from the LLM's full answer
def generate_summary(text, api_key):
    prompt = f"""
Summarize the following answer into 3–4 bullet points in plain English.

{text}
"""
    try:
        llm = ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo", openai_api_key=api_key)
        return llm.predict(prompt)
    except Exception as e:
        return f"Summary generation failed: {e}"

# Main Streamlit app function
def main():
    # Set basic app layout and title
    st.set_page_config(page_title="Automated Scheme Research Tool", layout="centered")
    st.title("🧾 Automated Scheme Research Tool")

    # Initialize session state variables to persist across interactions
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar section for input configuration
    with st.sidebar:
        st.header("📂 Input")
        input_mode = st.radio("Choose Input Type", ["Enter URLs", "Upload File"])
        urls_input = st.text_area("Paste URLs (one per line):") if input_mode == "Enter URLs" else ""
        uploaded_file = st.file_uploader("Upload a PDF or TXT file") if input_mode == "Upload File" else None
        process_btn = st.button("📥 Process")    # Process the input
        clear_btn = st.button("♻️ Clear All")    # Reset everything

    # Main input box for user query
    query = st.text_input("💬 Ask a question:")
    submit_btn = st.button("🤖 Get Answer")

    # Clear/reset everything on button click
    if clear_btn:
        st.session_state.vectorstore = None
        st.session_state.processed = False
        st.session_state.chat_history = []
        if os.path.exists("faiss_store_openai"):
            shutil.rmtree("faiss_store_openai")
        st.rerun()

    # Processing logic for either uploaded file or URL input
    if process_btn:
        try:
            api_key = load_api_key()
            openai.api_key = api_key

            # Reset previous processing results
            st.session_state.vectorstore = None
            st.session_state.processed = False

            # Delete previous FAISS index if it exists
            if os.path.exists("faiss_store_openai"):
                shutil.rmtree("faiss_store_openai")

            # Handle file upload processing
            if input_mode == "Upload File" and uploaded_file:
                raw_text = extract_text_from_file(uploaded_file)
                cleaned = clean_text(raw_text)
                if not cleaned.strip():
                    st.error("No readable content found in the file.")
                else:
                    with st.spinner("🔄 Processing uploaded file..."):
                        docs = load_and_split_documents(file_content=cleaned, file_name=uploaded_file.name)
                        if docs:
                            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                            vs = create_or_load_faiss_index(docs, embeddings)
                            st.session_state.vectorstore = vs
                            st.session_state.processed = True
                        else:
                            st.error("Failed to process file content.")

            # Handle URL input processing
            elif input_mode == "Enter URLs":
                url_list = [u.strip() for u in urls_input.split("\n") if u.strip()]
                if not url_list:
                    st.warning("Please enter at least one valid URL.")
                else:
                    with st.spinner("🔄 Processing URLs..."):
                        docs = load_and_split_documents(url_list=url_list)
                        if docs:
                            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                            vs = create_or_load_faiss_index(docs, embeddings)
                            st.session_state.vectorstore = vs
                            st.session_state.processed = True
                        else:
                            st.error("Could not extract useful content from the URLs.")
        except Exception as e:
            st.error(f"Processing error: {e}")

    # Handle user query after data has been processed
    if submit_btn:
        try:
            api_key = load_api_key()
            openai.api_key = api_key

            if not query.strip():
                st.warning("Enter a question to ask.")
            elif not st.session_state.vectorstore:
                st.warning("Process a file or URLs first.")
            else:
                with st.spinner("🤖 Generating answer..."):
                    answer, sources = query_llm(
                        st.session_state.vectorstore, query, model_name="gpt-3.5-turbo", k=5
                    )
                    summary = generate_summary(answer, api_key)
                    # Save Q&A + summary and sources to chat history
                    st.session_state.chat_history.append((query, answer, summary, sources))
        except Exception as e:
            st.error(f"Answering error: {e}")

    # Display previous questions, answers, summaries and sources
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for i, (q, a, s, srcs) in enumerate(st.session_state.chat_history, 1):
            with st.expander(f"Q{i}: {q}", expanded=True):
                st.markdown(f"**Answer:**\n\n{a}")
                st.markdown(f"**Summary:**\n\n{s}")
                if srcs:
                    st.markdown("**Sources:**")
                    for src in sorted(set(srcs)):
                        st.markdown(f"- [{src}]({src})")

# Run the app
if __name__ == "__main__":
    main()
