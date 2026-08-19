import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()
import os
api_key = os.getenv("GOOGLE_API_KEY")

st.title("PDF QnA SYSTEM")

with st.sidebar:
    uploaded_file = st.file_uploader("upload pdf",type = ['pdf'])

    if uploaded_file:
        with open("temp.pdf","wb") as f:
            f.write(uploaded_file.getbuffer())  

        loader = PyPDFLoader("temp.pdf")  
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200
        )

        chunks = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_db = Chroma.from_documents(
            documents = chunks,
            embedding = embeddings
        )


    

question = st.text_input("write your query here")

if question and uploaded_file:
    retriever = vector_db.as_retriever(
    search_kwargs= {"k":3}
)
    relevant_docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in relevant_docs
    )

    llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    api_key = api_key,
    temperature=0
)
    prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{question}

If the answer is not present in the context,
say "I don't know based on the provided PDF."
"""
    response = llm.invoke(prompt)

    st.write(response.text)
    

        



            



