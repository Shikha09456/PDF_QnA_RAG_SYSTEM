import streamlit as st
import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")


# --------------------------------------------------
# 2. Page
# --------------------------------------------------

st.title("📄 PDF QnA SYSTEM")


# --------------------------------------------------
# 3. Session State
# --------------------------------------------------

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None
if "messages" not in st.session_state:
    st.session_state.messages = []    


# --------------------------------------------------
# 4. Cache Embedding Model
# --------------------------------------------------

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# --------------------------------------------------
# 5. Cache LLM
# --------------------------------------------------

@st.cache_resource
def get_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        api_key=api_key,
        temperature=0
    )


# --------------------------------------------------
# 6. Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("Upload PDF")

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"]
    )


    # --------------------------------------------------
    # 7. Process PDF
    # --------------------------------------------------

    if uploaded_file:

        if st.button("Process PDF"):

            with st.spinner("Processing PDF..."):

                # Save uploaded PDF temporarily
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())


                # Load PDF
                loader = PyPDFLoader("temp.pdf")

                documents = loader.load()


                # Split text
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150
                )

                chunks = text_splitter.split_documents(documents)


                # Get cached embedding model
                embeddings = get_embeddings()


                # Create vector database
                vector_db = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings
                )


                # Store vector DB in session
                st.session_state.vector_db = vector_db

                # Remember uploaded file
                st.session_state.file_name = uploaded_file.name


            st.success("PDF processed successfully! 🎉")


# --------------------------------------------------
# 8. Show currently processed PDF
# --------------------------------------------------

if st.session_state.vector_db is not None:

    st.info(
        f"Currently loaded PDF: "
        f"{st.session_state.file_name}"
    )

# --------------------------------------------------
# 8.5 Chat History
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])






# --------------------------------------------------
# 9. Question
# --------------------------------------------------

question = st.chat_input("Ask a question about your PDF")




# --------------------------------------------------
# 10. QnA
# --------------------------------------------------

if question:

    st.session_state.messages.append({
            "role": "user",
            "content": question
        })
 
    history = st.session_state.messages[:-1]

    conversation = "\n".join(
                f"{message['role']}: {message['content']}"
                for message in history
            )  
        
        
    with st.chat_message("user"):
                    st.write(question)
                    
    if st.session_state.vector_db is None:
        st.warning("Please upload and process a PDF first.")    
    

    else:

        # Get vector DB from session
        vector_db = st.session_state.vector_db


        # --------------------------------------------------
        # Retrieval
        # --------------------------------------------------

        with st.spinner("Searching PDF..."):

            retriever = vector_db.as_retriever(
                search_kwargs={"k": 3}
            )

            relevant_docs = retriever.invoke(question)


        # --------------------------------------------------
        # Create context
        # --------------------------------------------------

        context = "\n\n".join(
            doc.page_content
            for doc in relevant_docs
        )


        # --------------------------------------------------
        # Get cached LLM
        # --------------------------------------------------

        llm = get_llm()


        # --------------------------------------------------
        # Prompt
        # --------------------------------------------------

        prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{question}

Conversation History:
{conversation}

If the answer is not present in the context,
say:

"I don't know based on the provided PDF."
"""


        # --------------------------------------------------
        # LLM
        # --------------------------------------------------

        with st.spinner("Generating answer..."):

            response = llm.invoke(prompt)


        # --------------------------------------------------
        # Display answer
        # --------------------------------------------------

        st.session_state.messages.append({
    "role": "assistant",
    "content": response.text
})

        with st.chat_message("assistant"):
            st.write(response.text)