import os
import pickle
import re

import faiss
import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


VECTOR_FOLDER = "vectorstore"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"

TOP_K = 5


print("Loading health cover knowledge base...")

# Load FAISS index
index = faiss.read_index(
    os.path.join(VECTOR_FOLDER, "health_index.faiss")
)

# Load stored chunks
with open(
    os.path.join(VECTOR_FOLDER, "chunks.pkl"),
    "rb"
) as file:
    documents = pickle.load(file)


# Load embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)


# =========================================================
# BM25 SETUP
# =========================================================

def tokenize(text):
    """
    Simple tokenizer used only for BM25.
    """

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


# Tokenise the same chunks that are already
# being used by the FAISS system.
tokenized_documents = [
    tokenize(document["text"])
    for document in documents
]


# Build BM25 index
bm25_index = BM25Okapi(
    tokenized_documents
)


# =========================================================
#  DENSE FAISS RETRIEVAL
# =========================================================

def dense_retrieve(question, top_k=4):
    """
    Retrieve the most relevant document chunks
    using the original Dense FAISS retrieval.
    """

    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    ).astype("float32")

    # Retrieve extra results so we can filter by insurer if needed
    search_k = min(30, index.ntotal)

    scores, indices = index.search(
        question_embedding,
        search_k
    )

    question_lower = question.lower()

    required_source = None

    if "medibank" in question_lower:
        required_source = "medibank"

    elif "bupa" in question_lower:
        required_source = "bupa"

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        document = documents[idx]

        source_lower = document["source"].lower()

        # If user asks specifically about one insurer,
        # don't provide evidence from another insurer.
        if required_source:

            if required_source == "medibank":

                if (
                    "essentials" not in source_lower
                    and "medibank" not in source_lower
                ):
                    continue

            elif required_source == "bupa":

                if "bupa" not in source_lower:
                    continue

        results.append(
            {
                "text": document["text"],
                "source": document["source"],
                "page": document["page"],
                "score": float(score)
            }
        )

        if len(results) >= top_k:
            break

    return results


# =========================================================
# BM25 RETRIEVAL
# =========================================================

def bm25_retrieve(question, top_k=4):
    """
    Retrieve the most relevant document chunks
    using BM25 keyword retrieval.
    """

    query_tokens = tokenize(question)

    scores = bm25_index.get_scores(
        query_tokens
    )

    # Sort chunk indexes from highest BM25 score
    # to lowest BM25 score.
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    question_lower = question.lower()

    required_source = None

    if "medibank" in question_lower:
        required_source = "medibank"

    elif "bupa" in question_lower:
        required_source = "bupa"

    results = []

    for idx in ranked_indices:

        document = documents[idx]

        source_lower = document["source"].lower()

        # Same insurer filtering used in Dense retrieval
        if required_source:

            if required_source == "medibank":

                if (
                    "essentials" not in source_lower
                    and "medibank" not in source_lower
                ):
                    continue

            elif required_source == "bupa":

                if "bupa" not in source_lower:
                    continue

        results.append(
            {
                "text": document["text"],
                "source": document["source"],
                "page": document["page"],
                "score": float(scores[idx])
            }
        )

        if len(results) >= top_k:
            break

    return results


# =========================================================
# RETRIEVAL SELECTOR
# =========================================================

def retrieve(
    question,
    top_k=4,
    method="dense"
):
    """
    Choose which retrieval method to use.

    dense = original MiniLM + FAISS retrieval
    bm25  = BM25 keyword retrieval
    """

    if method == "bm25":

        return bm25_retrieve(
            question,
            top_k
        )

    # Dense remains the default so existing code
    # calling retrieve(question, top_k) still works.
    return dense_retrieve(
        question,
        top_k
    )


# =========================================================
# ORIGINAL ANSWER GENERATION
# =========================================================

def generate_answer(question, retrieved_chunks):

    if not retrieved_chunks:
        return (
            "I could not find enough information in the supplied "
            "health cover documents to answer this question."
        )

    context_parts = []

    for number, result in enumerate(
        retrieved_chunks,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {number}
Document: {result['source']}
Page: {result['page']}

{result['text']}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are a health cover information assistant.

Answer the user's question ONLY using the information contained
in the supplied document context.

Important rules:

1. Do not use outside knowledge.
2. Do not invent insurance benefits, prices, exclusions or waiting periods.
3. Clearly distinguish between different insurers.
4. If the supplied context does not contain enough information,
   say: "I could not find enough information in the supplied documents
   to answer this question."
5. Mention important limitations, waiting periods or possible
   out-of-pocket costs when relevant.
6. Keep the answer clear and concise.
7. Do not make up page numbers or sources.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.1
        }
    )

    return response["message"]["content"]


# =========================================================
# MAIN PROGRAM
# =========================================================

def main():

    print("\n========================================")
    print("     Health Cover RAG Assistant")
    print("========================================")

    # choose retrieval method
    print("\nChoose retrieval method:")
    print("1. Dense FAISS")
    print("2. BM25")

    choice = input(
        "\nEnter 1 or 2: "
    ).strip()

    if choice == "2":

        retrieval_method = "bm25"

        print(
            "\nUsing BM25 retrieval."
        )

    else:

        retrieval_method = "dense"

        print(
            "\nUsing Dense FAISS retrieval."
        )

    print("\nType 'exit' to close the program.\n")

    while True:

        question = input(
            "Ask a question: "
        ).strip()

        if question.lower() in [
            "exit",
            "quit"
        ]:

            print("\nGoodbye!")
            break

        if not question:
            continue

        print(
            "\nSearching health cover documents...\n"
        )

        retrieved_chunks = retrieve(
            question,
            TOP_K,
            method=retrieval_method
        )

        answer = generate_answer(
            question,
            retrieved_chunks
        )

        print("ANSWER")
        print("------")
        print(answer)

        if retrieved_chunks:

            print("\nSOURCES")
            print("-------")

            shown_sources = set()

            for result in retrieved_chunks:

                source_key = (
                    result["source"],
                    result["page"]
                )

                if source_key not in shown_sources:

                    print(
                        f"- {result['source']} "
                        f"(Page {result['page']})"
                    )

                    shown_sources.add(
                        source_key
                    )

        print(
            "\n========================================\n"
        )


if __name__ == "__main__":
    main()