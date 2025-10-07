import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
RAG_SOURCE_FILE = Path("./data/rag_corpus.txt")
CHROMA_PERSIST_DIR = Path("./chroma_db")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- 1. Load the Data ---
def load_data(file_path):
    """Loads the text file into a LangChain Document object."""
    print(f"Loading document from: {file_path}")
    if not file_path.exists():
        print(f"Error: RAG source file not found at {file_path}")
        return None
        
    loader = TextLoader(str(file_path))
    documents = loader.load()
    return documents

# --- 2. Split the Document into Chunks ---
def split_documents(documents):
    """Splits the loaded documents into smaller, coherent chunks."""
    print("Splitting document into manageable chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Max size of a chunk
        chunk_overlap=100, # Overlap helps maintain context across chunks
        length_function=len,
        separators=["\n\n", "\n", " ", ""] # Try to split on large breaks first
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks

# --- 3. Create Embeddings and Store in ChromaDB ---
def create_vector_store(chunks, persist_directory):
    """Converts chunks to vectors and stores them in ChromaDB."""
    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    print(f"Creating and persisting vector store to: {persist_directory}")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_directory)
    )
    vectorstore.persist()
    print("✅ RAG Indexing Complete!")

# --- Main Execution ---
if __name__ == "__main__":
    
    documents = load_data(RAG_SOURCE_FILE)
    if not documents:
        exit()

    chunks = split_documents(documents)
    create_vector_store(chunks, CHROMA_PERSIST_DIR)