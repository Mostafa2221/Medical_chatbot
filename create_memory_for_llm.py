from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from langchain_community.vectorstores import FAISS

def load_pdf_files(data):
    loader= DirectoryLoader(
        data,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

extracted_data= load_pdf_files("C:/Users/hawki/Desktop/Medical_chatbot/data")

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

text_chunks = create_chunks(extracted_data)
print(f"Number of chunks created: {len(text_chunks)}")

def get_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    return embeddings

embedding_model = get_embeddings()

DB_FAISS_PATH = "VectorStore/dbfaiss"
db= FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)