import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration (These are constants) ---
# ⚠️ WARNING: API key exposed in code - NOT recommended for production!
# Get your API key from: https://makersuite.google.com/app/apikey
GOOGLE_API_KEY = "INSERT_YOUR_API_KEY"  # Replace with your actual API key

VECTOR_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "models/gemini-embedding-001" 

def ingest_data_from_bytes(document_bytes: bytes, filename: str):
    """
    Ingests document data (passed as raw bytes) into a local ChromaDB.

    Args:
        document_bytes: The raw content of the file (e.g., from an uploaded file).
        filename: The original name of the file (for temporary file creation).
    """
    
    # 1. Temporarily save the bytes to a file so PyPDFLoader can read it
    # We use a temp directory to prevent cluttering the main project folder
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, filename)
        
        with open(temp_file_path, "wb") as tmp_file:
            tmp_file.write(document_bytes)

        print(f"Loading document: {filename}")
        
        # 2. Load Data
        # PyPDFLoader needs a file path, which we provide with the temporary path
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

    # 3. Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")
    
    # 4. Embed and Store
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )
    
    # Create the vector store. This will generate embeddings and save them to disk.
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory=VECTOR_DB_PATH
    )
    vectorstore.persist()
    print(f"Data ingestion complete. Vector store saved to {VECTOR_DB_PATH}")

# --- Example of running the script with a static file (for local testing) ---
if __name__ == "__main__":
    # --- IMPORTANT ---
    # To run this, you must have a test PDF in your project folder.
    STATIC_DOCUMENT_PATH = "test_document.pdf" # <<< CHANGE THIS TO YOUR FILE NAME
    
    if not os.path.exists(STATIC_DOCUMENT_PATH):
        print(f"Error: Static document not found at {STATIC_DOCUMENT_PATH}")
        print("Please create a 'test_document.pdf' or update the path.")
    else:
        print("--- Running Ingestion for Local File ---")
        
        # Read the file bytes
        with open(STATIC_DOCUMENT_PATH, "rb") as f:
            file_bytes = f.read()
        
        # Call the refactored function
        ingest_data_from_bytes(file_bytes, os.path.basename(STATIC_DOCUMENT_PATH))

        print("--- Local Ingestion Finished ---")
