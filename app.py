import streamlit as st
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import pipeline

# ---------------- PAGE CONFIG ----------------

st.set_page_config(page_title="Enterprise Free RAG")

st.title("📄 Enterprise-Level FREE RAG Chatbot")

st.write("Upload PDFs and ask questions from your documents.")

# ---------------- FILE UPLOAD ----------------

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

# ---------------- PROCESS PDF ----------------

if uploaded_files:

    all_text = ""

    for uploaded_file in uploaded_files:

        pdf_reader = PdfReader(uploaded_file)

        for page in pdf_reader.pages:

            text = page.extract_text()

            if text:
                all_text += text

    # ---------------- TEXT CHUNKING ----------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(all_text)

    st.success(f"✅ Total Chunks Created: {len(chunks)}")

    # ---------------- EMBEDDINGS ----------------

    with st.spinner("Creating embeddings..."):

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        db = FAISS.from_texts(chunks, embeddings)

    st.success("✅ FAISS Vector Database Created")

    # ---------------- LOAD MODEL ----------------

    st.write("Loading AI model... First time may take few minutes.")

    @st.cache_resource
    def load_model():

        pipe = pipeline(
            task="text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            max_new_tokens=256
        )

        return pipe

    llm = load_model()

    st.success("✅ AI Model Loaded")

    # ---------------- QUESTION INPUT ----------------

    question = st.text_input("Ask your PDFs")

    # ---------------- ASK QUESTION ----------------

    if question:

        with st.spinner("Searching documents..."):

            docs = db.similarity_search(question, k=4)

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            prompt = f"""
You are an AI assistant.

Answer ONLY from the provided context.

If answer is not available, say:
"I could not find the answer in the document."

Context:
{context}

Question:
{question}

Answer:
"""

            result = llm(prompt)

            answer = result[0]["generated_text"]

        # ---------------- OUTPUT ----------------

        st.subheader("🤖 AI Answer")

        st.write(answer)

        # ---------------- SOURCES ----------------

        with st.expander("📚 Retrieved Sources"):

            for i, doc in enumerate(docs):

                st.write(f"### Source {i+1}")

                st.write(doc.page_content[:1000])