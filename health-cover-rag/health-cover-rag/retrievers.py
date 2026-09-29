import pickle
import re

import faiss
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# =========================================================
# FILES
# =========================================================

INDEX_PATH = "vectorstore/health_index.faiss"
CHUNKS_PATH = "vectorstore/chunks.pkl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# =========================================================
# LOAD CHUNKS
# =========================================================

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)


# =========================================================
# LOAD DENSE FAISS
# =========================================================

faiss_index = faiss.read_index(INDEX_PATH)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


# =========================================================
# BM25 SETUP
# =========================================================

def tokenize(text):
    """
    Simple tokenizer for BM25.
    Converts text to lowercase words.
    """

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


tokenized_corpus = [
    tokenize(chunk["text"])
    for chunk in chunks
]

bm25 = BM25Okapi(
    tokenized_corpus
)


# =========================================================
# OPTIONAL INSURER FILTER
# =========================================================

def detect_insurer(question):

    q = question.lower()

    if "medibank" in q:
        return "medibank"

    if "bupa" in q:
        return "bupa"

    if "nib" in q:
        return "nib"

    return None


def source_matches_insurer(source, insurer):

    if insurer is None:
        return True

    return insurer.lower() in source.lower()


# =========================================================
# DENSE FAISS RETRIEVAL
# =========================================================

def dense_retrieve(question, top_k=5):

    insurer = detect_insurer(question)

    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )

    # Retrieve more candidates before filtering
    candidate_k = min(
        30,
        faiss_index.ntotal
    )

    scores, indices = faiss_index.search(
        question_embedding,
        candidate_k
    )

    results = []

    for score, index_id in zip(
        scores[0],
        indices[0]
    ):

        if index_id == -1:
            continue

        chunk = chunks[index_id]

        source = chunk["source"]

        if not source_matches_insurer(
            source,
            insurer
        ):
            continue

        results.append(
            {
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "score": float(score),
                "retriever": "dense"
            }
        )

        if len(results) >= top_k:
            break

    return results


# =========================================================
# BM25 RETRIEVAL
# =========================================================

def bm25_retrieve(question, top_k=5):

    insurer = detect_insurer(question)

    query_tokens = tokenize(question)

    scores = bm25.get_scores(
        query_tokens
    )

    # Highest score first
    ranked_indices = np.argsort(
        scores
    )[::-1]

    results = []

    for index_id in ranked_indices:

        chunk = chunks[index_id]

        source = chunk["source"]

        if not source_matches_insurer(
            source,
            insurer
        ):
            continue

        results.append(
            {
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "score": float(
                    scores[index_id]
                ),
                "retriever": "bm25"
            }
        )

        if len(results) >= top_k:
            break

    return results


# =========================================================
# GENERAL RETRIEVER
# =========================================================

def retrieve(
    question,
    top_k=5,
    method="dense"
):

    if method == "dense":

        return dense_retrieve(
            question,
            top_k
        )

    elif method == "bm25":

        return bm25_retrieve(
            question,
            top_k
        )

    else:

        raise ValueError(
            f"Unknown retrieval method: {method}"
        )