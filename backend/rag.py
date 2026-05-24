import uuid
import logging

import chromadb

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from transformers import pipeline

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# =========================================================
# LOAD MODELS
# =========================================================

embedder = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

# =========================================================
# CHROMADB
# =========================================================

chroma = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma.get_or_create_collection(
    name="documents"
)

# =========================================================
# EMBEDDING
# =========================================================

def generate_embedding(text):

    return embedder.encode(
        [text],
        normalize_embeddings=True
    ).tolist()

# =========================================================
# ADD DOCUMENT
# =========================================================

def add_document(user, text):

    if not text.strip():
        return

    chunk_size = 1000

    chunks = [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
    ]

    for chunk in chunks:

        embedding = generate_embedding(chunk)[0]

        collection.add(
            ids=[str(uuid.uuid4())],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{
                "user": user
            }]
        )

# =========================================================
# PROMPT
# =========================================================

def build_prompt(query, context):

    return f"""
You are an advanced AI assistant.

Answer ONLY using the provided context.

If answer is unavailable, say:
"I could not find that information."

================ CONTEXT ================

{context}

================ QUESTION ================

{query}

================ RESPONSE ================
"""

# =========================================================
# RETRIEVE ANSWER
# =========================================================

def retrieve_answer(user, query):

    try:

        query_embedding = generate_embedding(query)

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=10,
            where={"user": user}
        )

        docs = results["documents"][0]

        if not docs:

            return {
                "answer": "No relevant documents found.",
                "sources": []
            }

        scores = reranker.predict([
            (query, d)
            for d in docs
        ])

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        top_docs = [
            d[0]
            for d in ranked[:4]
        ]

        context = "\n\n".join(top_docs)

        prompt = build_prompt(
            query,
            context
        )

        response = llm(
            prompt,
            max_new_tokens=256,
            do_sample=False
        )[0]["generated_text"]

        return {
            "answer": response,
            "sources": top_docs
        }

    except Exception as e:

        logger.error(str(e))

        return {
            "answer": f"RAG Error: {str(e)}",
            "sources": []
        }