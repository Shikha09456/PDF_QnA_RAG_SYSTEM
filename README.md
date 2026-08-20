Bilkul — tumhare GitHub project ke liye ek **professional but beginner-friendly README** ye rahega:

# PDF QnA System using RAG

A simple **PDF Question Answering system** built using **Retrieval-Augmented Generation (RAG)**. The application allows users to upload a PDF and ask questions based on its content.

The system retrieves the most relevant sections from the uploaded PDF and uses an LLM to generate an answer based only on the retrieved context.

## 🚀 Features

* Upload a PDF directly through the Streamlit interface
* Extract text from PDF documents
* Split large documents into smaller chunks
* Generate embeddings using HuggingFace
* Store document embeddings in ChromaDB
* Perform semantic similarity search
* Retrieve the most relevant chunks for a question
* Generate answers using Google Gemini
* Uses Streamlit session state to maintain the vector database
* Uses caching to reduce unnecessary model loading

## 🏗️ RAG Architecture

```text
                PDF Upload
                    │
                    ▼
              PyPDFLoader
                    │
                    ▼
             Text Extraction
                    │
                    ▼
            Text Chunking
                    │
                    ▼
         HuggingFace Embeddings
                    │
                    ▼
              ChromaDB
            Vector Database
                    │
                    │
             User Question
                    │
                    ▼
               Retriever
                    │
                    ▼
          Relevant PDF Chunks
                    │
                    ▼
              Gemini LLM
                    │
                    ▼
                Answer
```

## 🛠️ Tech Stack

* **Python**
* **Streamlit** — Web application
* **LangChain** — RAG pipeline
* **PyPDF** — PDF text extraction
* **HuggingFace Sentence Transformers** — Text embeddings
* **ChromaDB** — Vector database
* **Google Gemini** — Large Language Model

## 📂 Project Structure

```text
pdf-qna-rag/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env
```

> ⚠️ `.env` should **not** be uploaded to GitHub because it contains your API key.

## ⚙️ How It Works

### 1. Upload PDF

The user uploads a PDF using Streamlit's file uploader.

### 2. Extract Text

`PyPDFLoader` extracts the text from the PDF.

### 3. Split Text

The extracted text is divided into smaller chunks using:

```python
RecursiveCharacterTextSplitter
```

Example configuration:

```python
chunk_size = 1000
chunk_overlap = 100
```

### 4. Generate Embeddings

Each text chunk is converted into a numerical vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

### 5. Store in ChromaDB

The generated embeddings are stored in ChromaDB so that relevant information can be retrieved efficiently.

### 6. Retrieve Relevant Information

When the user asks a question, the retriever performs semantic similarity search and retrieves the most relevant chunks.

### 7. Generate Answer

The retrieved chunks are passed as context to the Gemini model.

The model is instructed to answer using only the provided PDF context.

If the answer cannot be found in the PDF, the application responds:

```text
I don't know based on the provided PDF.
```

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

Go into the project directory:

```bash
cd pdf-qna-rag
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🔑 API Key Setup

Create a `.env` file:

```text
GOOGLE_API_KEY=your_google_api_key
```

Make sure `.env` is included in `.gitignore`:

```text
.env
```

## ▶️ Run the Application

Run:

```bash
streamlit run app.py
```

The application will open in your browser.

## ⚡ Performance Optimization

The application uses Streamlit caching to avoid repeatedly loading expensive resources.

For example:

```python
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
```

The vector database is maintained using Streamlit session state:

```python
st.session_state.vector_db
```

This prevents the PDF from being re-indexed every time the user asks a new question.

## 📚 What I Learned

Through this project, I learned:

* How RAG works
* PDF document loading
* Document chunking
* Text embeddings
* Semantic search
* Vector databases
* ChromaDB
* LangChain basics
* Retriever implementation
* LLM integration
* Prompt construction
* Streamlit application development
* Streamlit caching
* Streamlit session state
* Git and GitHub project management

## 🔮 Future Improvements

Possible improvements include:

* [ ] Add chat history
* [ ] Support multiple PDFs
* [ ] Add PDF page references to answers
* [ ] Improve retrieval using a reranker
* [ ] Add hybrid search
* [ ] Add streaming responses
* [ ] Add better error handling
* [ ] Improve UI/UX
* [ ] Add persistent vector database
* [ ] Deploy the application on Streamlit Community Cloud

## 🎯 Project Goal

The goal of this project is to understand and implement a complete **Retrieval-Augmented Generation pipeline** from document ingestion to retrieval and LLM-based answer generation.

## 👩‍💻 Author

**Shikha Kumari**

Built as a learning project to understand **RAG, LangChain, vector databases, embeddings, LLMs, and Streamlit**.
