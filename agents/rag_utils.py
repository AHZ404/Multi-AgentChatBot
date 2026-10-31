import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# 1. Initialize Free LLM
def get_llm():
    return ChatGroq(model_name="openai/gpt-oss-20b", temperature=0.2)


# 2. Build or Load Vector Store Retriever for RAG
def get_rag_retriever(doc_subfolder: str):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store_path = f"faiss_index_{doc_subfolder}"

    if os.path.exists(vector_store_path):
        vectorstore = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    else:
        # Load markdown/text files from docs/<subfolder>
        loader = DirectoryLoader(f"docs/{doc_subfolder}", glob="**/*.txt", loader_cls=TextLoader)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local(vector_store_path)

    return vectorstore.as_retriever(search_kwargs={"k": 3})