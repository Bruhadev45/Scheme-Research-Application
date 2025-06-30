import configparser
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

def load_api_key():
    config = configparser.ConfigParser()
    config.read('.config')
    return config['OPENAI']['api_key']

def load_and_split_documents(url_list):
    try:
        loader = UnstructuredURLLoader(urls=url_list)
        docs = loader.load()
        if not docs:
            return []
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        return splitter.split_documents(docs)
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []

def create_or_load_faiss_index(docs, embeddings, index_path='faiss_store_openai'):
    try:
        if os.path.exists(index_path):
            return FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(index_path)
        return vectorstore
    except Exception as e:
        print(f"Error in FAISS index creation/loading: {e}")
        return None

def query_llm(vectorstore, query, k=4):
    try:
        relevant_docs = vectorstore.similarity_search(query, k=k)
        if not relevant_docs:
            return "No relevant documents found.", []

        sources = [doc.metadata.get("source", "N/A") for doc in relevant_docs]
        content = "\n\n".join([doc.page_content for doc in relevant_docs])

        api_key = load_api_key()
        llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            openai_api_key=api_key
        )

        prompt = (
            f"Use only the following content to answer the question:\n\n{content}\n\n"
            f"Question: {query}\n\n"
            "Answer clearly and concisely. If the content does not contain the answer, say: "
            "'I don't know based on the provided information.' Do not add information that is not present in the text."
        )

        response = llm.invoke(prompt)
        return response, sources
    except Exception as e:
        print(f"Error during LLM query: {e}")
        return f"Failed to generate answer: {e}", []
