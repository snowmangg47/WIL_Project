import os
import pickle
import re

import faiss
import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)


VECTOR_FOLDER = "vectorstore"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K = 5

CANDIDATE_K = 20


print("Loading health cover knowledge base...")

# Load FAISS index
index = faiss.read_index(
    os.path.join(
        VECTOR_FOLDER,
        "health_index.faiss"
    )
)

# Load stored chunks
with open(
    os.path.join(
        VECTOR_FOLDER,
        "chunks.pkl"
    ),
    "rb"
) as file:

    documents = pickle.load(file)


# Load embedding model
embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


# Load reranker
print("Loading reranker...")

try:

    reranker_model = CrossEncoder(
        RERANKER_MODEL
    )

except Exception as error:

    print(
        "Warning: reranker could not be loaded."
    )

    print(
        error
    )

    reranker_model = None


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
    tokenize(
        document["text"]
    )
    for document in documents
]


# Build BM25 index
bm25_index = BM25Okapi(
    tokenized_documents
)


# =========================================================
# INSURER HELPERS
# =========================================================

INSURER_ALIASES = {

    "medibank": [
        "medibank",
        "essentials"
    ],

    "bupa": [
        "bupa"
    ],

    "nib": [
        "nib"
    ]
}


def get_insurer_from_source(source):

    source_lower = source.lower()

    for insurer, aliases in (
        INSURER_ALIASES.items()
    ):

        for alias in aliases:

            if alias in source_lower:
                return insurer

    return None


def source_matches_insurer(
    source,
    insurer
):

    detected_insurer = (
        get_insurer_from_source(
            source
        )
    )

    return (
        detected_insurer
        == insurer
    )


def detect_insurers(question):

    question_lower = (
        question.lower()
    )

    insurers = []

    for insurer, aliases in (
        INSURER_ALIASES.items()
    ):

        found = False

        for alias in aliases:

            if alias in question_lower:

                found = True
                break

        if found:

            insurers.append(
                insurer
            )

    return insurers


# Find which insurers actually exist
# in the current knowledge base.
AVAILABLE_INSURERS = []

for document in documents:

    insurer = get_insurer_from_source(
        document["source"]
    )

    if (
        insurer
        and insurer
        not in AVAILABLE_INSURERS
    ):

        AVAILABLE_INSURERS.append(
            insurer
        )


def is_comparison_question(question):

    question_lower = (
        question.lower()
    )

    mentioned_insurers = (
        detect_insurers(
            question
        )
    )

    # If two or more insurers are named,
    # treat it as a comparison.
    if len(
        mentioned_insurers
    ) >= 2:

        return True


    comparison_terms = [

        "compare",
        "comparison",
        "versus",
        " vs ",
        "better",
        "best",
        "cheaper",
        "cheapest",
        "difference",
        "differences",
        "which cover",
        "which policy",
        "which insurer",
        "more affordable",
        "less expensive",
        "more suitable"
    ]


    return any(
        term in question_lower
        for term in comparison_terms
    )


def get_target_insurers(question):

    mentioned_insurers = (
        detect_insurers(
            question
        )
    )

    # If insurers were named,
    # retrieve information about those insurers.
    if mentioned_insurers:

        return mentioned_insurers


    # For a general comparison question,
    # use all recognised insurers in the KB.
    if is_comparison_question(
        question
    ):

        return (
            AVAILABLE_INSURERS.copy()
        )


    return []


def balance_results(
    results,
    insurers,
    top_k
):

    if (
        len(insurers) <= 1
        or not results
    ):

        return results[:top_k]


    buckets = {}

    for insurer in insurers:

        buckets[insurer] = [

            result
            for result in results

            if source_matches_insurer(
                result["source"],
                insurer
            )
        ]


    selected = []

    selected_keys = set()


    while len(selected) < top_k:

        added_result = False

        for insurer in insurers:

            if not buckets[
                insurer
            ]:

                continue


            result = buckets[
                insurer
            ].pop(0)


            result_key = (
                result["source"],
                result["page"],
                result["text"]
            )


            if (
                result_key
                in selected_keys
            ):

                continue


            selected.append(
                result
            )

            selected_keys.add(
                result_key
            )

            added_result = True


            if len(
                selected
            ) >= top_k:

                break


        if not added_result:
            break


    # If there are still spaces available,
    # fill them with the next strongest results.
    if len(selected) < top_k:

        for result in results:

            result_key = (
                result["source"],
                result["page"],
                result["text"]
            )

            if (
                result_key
                in selected_keys
            ):

                continue


            selected.append(
                result
            )

            selected_keys.add(
                result_key
            )


            if len(
                selected
            ) >= top_k:

                break


    return selected[:top_k]


# =========================================================
# DENSE FAISS RETRIEVAL
# =========================================================

def dense_retrieve(
    question,
    top_k=4
):
    """
    Retrieve the most relevant document chunks
    using Dense FAISS retrieval.
    """

    question_embedding = (
        embedding_model.encode(
            [question],
            normalize_embeddings=True
        ).astype(
            "float32"
        )
    )


    target_insurers = (
        get_target_insurers(
            question
        )
    )


    comparison = (
        is_comparison_question(
            question
        )
    )


    # For comparisons search the full index so that
    # one insurer does not dominate the candidate set.
    if (
        comparison
        and len(
            target_insurers
        ) > 1
    ):

        search_k = (
            index.ntotal
        )

    else:

        search_k = min(
            max(
                30,
                top_k * 3
            ),
            index.ntotal
        )


    scores, indices = (
        index.search(
            question_embedding,
            search_k
        )
    )


    results = []


    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:
            continue


        document = (
            documents[idx]
        )


        if target_insurers:

            matches_target = any(

                source_matches_insurer(
                    document["source"],
                    insurer
                )

                for insurer
                in target_insurers
            )


            if not matches_target:
                continue


        results.append(
            {
                "text":
                    document["text"],

                "source":
                    document["source"],

                "page":
                    document["page"],

                "score":
                    float(score)
            }
        )


        # For normal questions we can stop early.
        if not (
            comparison
            and len(
                target_insurers
            ) > 1
        ):

            if len(
                results
            ) >= top_k:

                break


    if (
        comparison
        and len(
            target_insurers
        ) > 1
    ):

        return balance_results(
            results,
            target_insurers,
            top_k
        )


    return results[:top_k]


# =========================================================
# BM25 RETRIEVAL
# =========================================================

def bm25_retrieve(
    question,
    top_k=4
):
    """
    Retrieve the most relevant document chunks
    using BM25 keyword retrieval.
    """

    query_tokens = tokenize(
        question
    )


    scores = (
        bm25_index.get_scores(
            query_tokens
        )
    )


    ranked_indices = sorted(

        range(
            len(scores)
        ),

        key=lambda i:
            scores[i],

        reverse=True
    )


    target_insurers = (
        get_target_insurers(
            question
        )
    )


    comparison = (
        is_comparison_question(
            question
        )
    )


    results = []


    for idx in ranked_indices:

        document = (
            documents[idx]
        )


        if target_insurers:

            matches_target = any(

                source_matches_insurer(
                    document["source"],
                    insurer
                )

                for insurer
                in target_insurers
            )


            if not matches_target:
                continue


        results.append(
            {
                "text":
                    document["text"],

                "source":
                    document["source"],

                "page":
                    document["page"],

                "score":
                    float(
                        scores[idx]
                    )
            }
        )


        if not (
            comparison
            and len(
                target_insurers
            ) > 1
        ):

            if len(
                results
            ) >= top_k:

                break


    if (
        comparison
        and len(
            target_insurers
        ) > 1
    ):

        return balance_results(
            results,
            target_insurers,
            top_k
        )


    return results[:top_k]


# =========================================================
# HYBRID RETRIEVAL
# =========================================================

def hybrid_retrieve(
    question,
    top_k=4
):
    """
    Combine Dense FAISS and BM25 results
    using Reciprocal Rank Fusion.
    """

    candidate_k = max(
        CANDIDATE_K,
        top_k
    )


    dense_results = (
        dense_retrieve(
            question,
            candidate_k
        )
    )


    bm25_results = (
        bm25_retrieve(
            question,
            candidate_k
        )
    )


    fused_results = {}


    # Standard RRF constant.
    rrf_k = 60


    for rank, result in enumerate(
        dense_results,
        start=1
    ):

        key = (
            result["source"],
            result["page"],
            result["text"]
        )


        if key not in fused_results:

            fused_results[key] = {

                "text":
                    result["text"],

                "source":
                    result["source"],

                "page":
                    result["page"],

                "score":
                    0.0
            }


        fused_results[
            key
        ][
            "score"
        ] += (
            1
            / (
                rrf_k
                + rank
            )
        )


    for rank, result in enumerate(
        bm25_results,
        start=1
    ):

        key = (
            result["source"],
            result["page"],
            result["text"]
        )


        if key not in fused_results:

            fused_results[key] = {

                "text":
                    result["text"],

                "source":
                    result["source"],

                "page":
                    result["page"],

                "score":
                    0.0
            }


        fused_results[
            key
        ][
            "score"
        ] += (
            1
            / (
                rrf_k
                + rank
            )
        )


    ranked_results = sorted(

        fused_results.values(),

        key=lambda x:
            x["score"],

        reverse=True
    )


    return ranked_results[
        :candidate_k
    ]


# =========================================================
# RERANKING
# =========================================================

def rerank_results(
    question,
    results,
    top_k
):
    """
    Rerank candidate passages using a
    Cross-Encoder model.
    """

    if not results:
        return []


    target_insurers = (
        get_target_insurers(
            question
        )
    )


    comparison = (
        is_comparison_question(
            question
        )
    )


    # If reranker failed to load,
    # continue using the original ranking.
    if reranker_model is None:

        ranked_results = (
            results.copy()
        )

    else:

        pairs = [

            [
                question,
                result["text"]
            ]

            for result in results
        ]


        rerank_scores = (
            reranker_model.predict(
                pairs
            )
        )


        ranked_order = sorted(

            range(
                len(results)
            ),

            key=lambda i:
                float(
                    rerank_scores[i]
                ),

            reverse=True
        )


        ranked_results = []


        for idx in ranked_order:

            result = dict(
                results[idx]
            )


            result[
                "retrieval_score"
            ] = result[
                "score"
            ]


            result[
                "score"
            ] = float(
                rerank_scores[idx]
            )


            ranked_results.append(
                result
            )


    # Make sure comparison answers still contain
    # evidence from each requested insurer.
    if (
        comparison
        and len(
            target_insurers
        ) > 1
    ):

        return balance_results(
            ranked_results,
            target_insurers,
            top_k
        )


    return ranked_results[
        :top_k
    ]


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

    dense  = MiniLM + FAISS
    bm25   = BM25 keyword retrieval
    hybrid = Dense FAISS + BM25 using RRF

    All approaches are followed by reranking.
    """

    candidate_k = max(
        CANDIDATE_K,
        top_k * 4
    )


    if method == "bm25":

        candidates = (
            bm25_retrieve(
                question,
                candidate_k
            )
        )


    elif method == "hybrid":

        candidates = (
            hybrid_retrieve(
                question,
                candidate_k
            )
        )


    else:

        candidates = (
            dense_retrieve(
                question,
                candidate_k
            )
        )


    return rerank_results(
        question,
        candidates,
        top_k
    )


# =========================================================
# ANSWER GENERATION
# =========================================================

def generate_answer(
    question,
    retrieved_chunks
):

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


    context = "\n".join(
        context_parts
    )


    prompt = f"""
You are a health cover information assistant.

Answer the user's question ONLY using the information contained
in the supplied document context.

Important rules:

1. Do not use outside knowledge.

2. Do not invent insurance benefits, prices, exclusions,
   waiting periods or policy conditions.

3. Clearly distinguish between different insurers.

4. If the supplied context does not contain enough information,
   say:
   "I could not find enough information in the supplied documents
   to answer this question."

5. Mention important limitations, waiting periods or possible
   out-of-pocket costs when relevant.

6. Keep the answer clear and concise.

7. Do not make up page numbers or sources.

8. Every factual statement taken from the documents must include
   the source number that directly supports it, using this exact
   format:
   [Source 1]
   [Source 2]
   etc.

9. Only cite a source when that source directly supports the
   statement. Do not cite sources that were not used in the answer.

10. If the user asks to compare insurers or policies, explain the
    information for each relevant insurer separately before making
    a comparison.

11. If the user asks which option is "better" generally, do not
    claim that one insurer is better overall. Explain the differences
    supported by the documents and state that the better option
    depends on the user's priorities.

12. If the user gives a specific measurable criterion, such as
    which listed policy is cheaper for the same duration and
    membership type, you may state which has the lower listed value
    when the supplied documents contain comparable information.

13. For price comparisons, only compare equivalent membership types
    and policy durations. For example, compare a 12-month single
    policy with another 12-month single policy.

14. If comparable information is missing for one insurer, clearly
    state that a fair comparison cannot be made from the supplied
    documents.

15. Missing information does NOT mean that an insurer does not
    provide a benefit.

16. For comparison questions, make sure the conclusion follows only
    from the evidence provided for the insurers being compared.

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
                "role":
                    "user",

                "content":
                    prompt
            }
        ],

        options={
            "temperature":
                0.1
        }
    )


    return response[
        "message"
    ][
        "content"
    ]


# =========================================================
# CITED SOURCE HELPERS
# =========================================================

def get_cited_sources(
    answer,
    retrieved_chunks
):
    """
    Return only the chunks that were actually cited
    by the generated answer.
    """

    citation_groups = re.findall(

        r"\[(?:Source|Sources)\s+([^\]]+)\]",

        answer,

        flags=re.IGNORECASE
    )


    source_numbers = []


    for group in citation_groups:

        numbers = re.findall(
            r"\d+",
            group
        )

        for number in numbers:

            number = int(
                number
            )

            if (
                number
                not in source_numbers
            ):

                source_numbers.append(
                    number
                )


    cited_results = []

    shown_sources = set()


    for number in source_numbers:

        index_number = (
            number - 1
        )


        if (
            index_number < 0
            or index_number
            >= len(
                retrieved_chunks
            )
        ):

            continue


        result = (
            retrieved_chunks[
                index_number
            ]
        )


        source_key = (
            result["source"],
            result["page"]
        )


        # Avoid showing the same document page
        # multiple times.
        if (
            source_key
            in shown_sources
        ):

            continue


        cited_results.append(
            result
        )

        shown_sources.add(
            source_key
        )


    return cited_results


def remove_source_citations(
    answer
):
    """
    Remove source markers before calculating
    text similarity metrics.
    """

    cleaned_answer = re.sub(

        r"\[(?:Source|Sources)\s+[^\]]+\]",

        "",

        answer,

        flags=re.IGNORECASE
    )


    cleaned_answer = re.sub(
        r"\s+",
        " ",
        cleaned_answer
    )


    return cleaned_answer.strip()


# =========================================================
# MAIN PROGRAM
# =========================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "     Health Cover RAG Assistant"
    )

    print(
        "========================================"
    )


    print(
        "\nChoose retrieval method:"
    )

    print(
        "1. Dense FAISS"
    )

    print(
        "2. BM25"
    )

    print(
        "3. Hybrid"
    )


    choice = input(
        "\nEnter 1, 2 or 3: "
    ).strip()


    if choice == "2":

        retrieval_method = (
            "bm25"
        )

        print(
            "\nUsing BM25 retrieval with reranking."
        )


    elif choice == "3":

        retrieval_method = (
            "hybrid"
        )

        print(
            "\nUsing Hybrid retrieval with reranking."
        )


    else:

        retrieval_method = (
            "dense"
        )

        print(
            "\nUsing Dense FAISS retrieval with reranking."
        )


    print(
        "\nType 'exit' to close the program.\n"
    )


    while True:

        question = input(
            "Ask a question: "
        ).strip()


        if question.lower() in [
            "exit",
            "quit"
        ]:

            print(
                "\nGoodbye!"
            )

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


        print(
            "ANSWER"
        )

        print(
            "------"
        )

        print(
            answer
        )


        cited_sources = (
            get_cited_sources(
                answer,
                retrieved_chunks
            )
        )


        if cited_sources:

            print(
                "\nSOURCES USED"
            )

            print(
                "------------"
            )


            for result in cited_sources:

                print(
                    f"- {result['source']} "
                    f"(Page {result['page']})"
                )


        print(
            "\n========================================\n"
        )


if __name__ == "__main__":

    main()