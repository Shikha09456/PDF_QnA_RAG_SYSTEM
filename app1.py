import streamlit as st
import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from sentence_transformers import CrossEncoder


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PDF QnA System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF QnA System")
st.caption("RAG + Cross-Encoder Reranking")


# ============================================================
# 3. SESSION STATE
# ============================================================

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# 4. CACHE EMBEDDING MODEL
# ============================================================

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# 5. CACHE RERANKER
# ============================================================

@st.cache_resource
def load_reranker():

    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


reranker = load_reranker()


# ============================================================
# 6. CACHE LLM
# ============================================================

@st.cache_resource
def get_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        api_key=api_key,
        temperature=0
    )


# ============================================================
# 7. RERANKING FUNCTION
# ============================================================

def rerank_documents(query, documents, top_k=5):

    if not documents:
        return []

    # Create query-document pairs
    pairs = [
        [query, doc.page_content]
        for doc in documents
    ]

    # Get relevance scores
    scores = reranker.predict(pairs)

    # Combine documents with scores
    ranked_documents = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Return top-k documents
    return [
        doc
        for doc, score in ranked_documents[:top_k]
    ]


# ============================================================
# 8. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📤 Upload PDF")

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"]
    )

    # --------------------------------------------------------
    # Process PDF
    # --------------------------------------------------------

    if uploaded_file:

        if st.button(
            "Process PDF",
            use_container_width=True
        ):

            with st.spinner("Processing PDF..."):

                # ------------------------------------------------
                # Save uploaded PDF temporarily
                # ------------------------------------------------

                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # ------------------------------------------------
                # Load PDF
                # ------------------------------------------------

                loader = PyPDFLoader("temp.pdf")

                documents = loader.load()

                # ------------------------------------------------
                # Split PDF into chunks
                # ------------------------------------------------

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150
                )

                chunks = text_splitter.split_documents(
                    documents
                )

                # ------------------------------------------------
                # Get embeddings
                # ------------------------------------------------

                embeddings = get_embeddings()

                # ------------------------------------------------
                # Create Chroma Vector DB
                # ------------------------------------------------

                vector_db = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings
                )

                # ------------------------------------------------
                # Store vector DB in session state
                # ------------------------------------------------

                st.session_state.vector_db = vector_db

                st.session_state.file_name = (
                    uploaded_file.name
                )

                # Clear previous chat
                st.session_state.messages = []

            st.success(
                "PDF processed successfully! 🎉"
            )


# ============================================================
# 9. SHOW CURRENT PDF
# ============================================================

if st.session_state.vector_db is not None:

    st.info(
        f"📄 Currently loaded PDF: "
        f"{st.session_state.file_name}"
    )


# ============================================================
# 10. DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# ============================================================
# 11. CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your PDF"
)


# ============================================================
# 12. QnA PIPELINE
# ============================================================

if question:

    # --------------------------------------------------------
    # Save user question
    # --------------------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # --------------------------------------------------------
    # Get previous conversation history
    # --------------------------------------------------------

    history = st.session_state.messages[:-1]

    conversation = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in history
    )

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.write(question)

    # --------------------------------------------------------
    # Check if PDF exists
    # --------------------------------------------------------

    if st.session_state.vector_db is None:

        st.warning(
            "Please upload and process a PDF first."
        )

    else:

        # ----------------------------------------------------
        # Get Vector DB
        # ----------------------------------------------------

        vector_db = st.session_state.vector_db


        # ====================================================
        # 13. RETRIEVAL
        # ====================================================

        with st.spinner(
            "Searching and reranking PDF..."
        ):

            # -----------------------------------------------
            # Create retriever
            # -----------------------------------------------

            retriever = vector_db.as_retriever(
                search_kwargs={
                    "k": 15
                }
            )

            # -----------------------------------------------
            # First stage retrieval
            # -----------------------------------------------

            retrieved_docs = retriever.invoke(
                question
            )

            # -----------------------------------------------
            # Second stage: Reranking
            # -----------------------------------------------

            relevant_docs = rerank_documents(
                query=question,
                documents=retrieved_docs,
                top_k=5
            )


        # ====================================================
        # 14. CREATE CONTEXT
        # ====================================================

        context = "\n\n".join(
            doc.page_content
            for doc in relevant_docs
        )


        # ====================================================
        # 15. GET LLM
        # ====================================================

        llm = get_llm()


        # ====================================================
        # 16. PROMPT
        # ====================================================

        prompt = f"""
You are a helpful PDF question-answering assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context,
say exactly:

"I don't know based on the provided PDF."

Do not make up information.

-------------------------
CONTEXT
-------------------------

{context}

-------------------------
QUESTION
-------------------------

{question}

-------------------------
CONVERSATION HISTORY
-------------------------

{conversation}
"""


        # ====================================================
        # 17. GENERATE ANSWER
        # ====================================================

        with st.spinner(
            "Generating answer..."
        ):

            response = llm.invoke(prompt)


        # ====================================================
        # 18. SAVE ASSISTANT RESPONSE
        # ====================================================

        st.session_state.messages.append({
            "role": "assistant",
            "content": response.text
        })


        # ====================================================
        # 19. DISPLAY ANSWER
        # ====================================================

        with st.chat_message("assistant"):

            st.write(response.text)


        # ====================================================
        # 20. DISPLAY SOURCES
        # ====================================================

        with st.expander(
            "📚 Retrieved Sources"
        ):

            for i, doc in enumerate(
                relevant_docs,
                start=1
            ):

                st.markdown(
                    f"**Source {i}**"
                )

                st.write(
                    doc.page_content
                )

                st.caption(
                    f"Page: "
                    f"{doc.metadata.get('page', 'N/A')} "
                    f"| Source: "
                    f"{doc.metadata.get('source', 'N/A')}"
                )

                st.divider()