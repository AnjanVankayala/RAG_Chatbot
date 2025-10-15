import streamlit as st
import os
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
# Import utility for working with uploaded files (using a temporary path)
from io import BytesIO 

# --- Configuration --
# Get your API key from: https://makersuite.google.com/app/apikey
GOOGLE_API_KEY = "INSERT_YOUR_API_KEY"  # Replace with your actual API key

VECTOR_DB_PATH = "./chroma_db_uploaded" # Use a new path for clarity
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash" 

def ingest_and_setup_rag(uploaded_file):
    """Loads PDF, creates embeddings, and sets up the RAG chain."""
    
    # 1. Save uploaded file to a temporary location to be read by PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    try:
        # 2. Load Data and Split Text
        st.info("Reading and splitting document...")
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(documents)
        st.success(f"Document split into {len(docs)} chunks.")
        
        # 3. Embed and Store
        st.info("Creating embeddings and setting up vector store...")
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY
        )
        
        # Note: We are creating a new in-memory/temp Chroma instance each time a file is uploaded
        vectorstore = Chroma.from_documents(
            documents=docs, 
            embedding=embeddings, 
            persist_directory=None # Don't persist to disk for simple temp use
        )
        
        # 4. Setup RAG Chain
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        system_prompt = (
            "You are an expert Q&A assistant. Answer the user's question ONLY "
            "based on the following context. If the context does not contain "
            "the answer, state 'I do not have enough information to answer that question.'"
            "\n\nContext: {context}"
        )
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        document_chain = create_stuff_documents_chain(llm, rag_prompt)
        rag_chain = create_retrieval_chain(retriever, document_chain)
        st.success("RAG Chatbot is ready!")
        
        # Return the chain and the retriever for use in chat
        return rag_chain, retriever

    finally:
        # 5. Clean up the temporary file
        os.remove(tmp_file_path)


# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="RAG Chatbot", layout="centered")
    st.title("RAG Chatbot")
    
    # --- File Uploader Sidebar ---
    with st.sidebar:
        st.header("Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file to process:", 
            type="pdf"
        )
        
        # Button to re-process and setup the RAG chain
        process_button = st.button("Process Document")

    # --- Main Chat Area ---
    if process_button and uploaded_file is not None:
        # Clear previous session and process new file
        st.session_state.messages = []
        with st.status("Processing document...", expanded=True) as status:
            try:
                # Store the RAG chain in session state
                st.session_state.rag_chain, st.session_state.retriever = ingest_and_setup_rag(uploaded_file)
                status.update(label="Document processed!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                status.update(label="Processing failed.", state="error", expanded=False)
                if 'rag_chain' in st.session_state:
                    del st.session_state.rag_chain # Remove invalid chain

    # Display chat
    if "rag_chain" in st.session_state:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("Ask a question about the document..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and display assistant response
            with st.spinner("Retrieving and generating answer..."):
                rag_chain = st.session_state.rag_chain
                response = rag_chain.invoke({"input": prompt})
                answer = response["answer"]
                
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.info("Upload a PDF document using the sidebar and click 'Process Document' to begin.")


if __name__ == "__main__":

    main()
