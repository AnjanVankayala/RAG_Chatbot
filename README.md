# RAG Chatbot with Google Gemini

A simple RAG (Retrieval-Augmented Generation) chatbot that lets you upload PDF documents and ask questions about their content.

## What it is

This chatbot uses AI to answer questions based on documents you upload. Instead of just using pre-trained knowledge, it finds relevant information from your specific documents and generates answers based on that content.

## How it works

1. **Upload a PDF** - The system extracts text from your document
2. **Create chunks** - Text is split into smaller pieces for better processing  
3. **Generate embeddings** - Each chunk is converted to a numerical vector
4. **Store in database** - Vectors are stored in ChromaDB for fast searching
5. **Ask questions** - When you ask something, it finds the most relevant chunks
6. **Generate answer** - Google Gemini AI creates an answer using only the relevant document content

## Tech Stack

- **Frontend**: Streamlit
- **AI Model**: Google Gemini 2.5 Flash
- **Embeddings**: Google Gemini Embeddings
- **Vector Database**: ChromaDB
- **Framework**: LangChain
- **PDF Processing**: PyPDF

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your Google API key
Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey) and update it in both `app.py` and `ingest.py`:

```python
GOOGLE_API_KEY = "your_actual_api_key_here"
```

### 3. Run the app
```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` and start uploading PDFs!

## Usage

1. Upload a PDF file using the sidebar
2. Click "Process Document" to analyze it
3. Ask questions about the document content
4. Get answers based on the uploaded document only

## Files

- `app.py` - Main Streamlit web application
- `ingest.py` - Script for processing documents programmatically
- `requirements.txt` - Python dependencies
