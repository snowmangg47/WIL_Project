import math
import pandas as pd

from bert_score import score as bert_score
from rouge_score import rouge_scorer
from tabulate import tabulate

from rag import retrieve, generate_answer


EVALUATION_FILE = "evaluation_questions.csv"

# We compare the system using 1, 3 and 5 retrieved chunks.
CUTOFFS = [1, 3, 5]

# Retrieval approaches to compare
METHODS = [
    "dense",
    "bm25"
]


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def parse_gold_pages(value):
    """
    Converts CSV values such as:

    10       -> {10}
    7|10     -> {7, 10}
    empty    -> set()
    """

    if pd.isna(value):
        return set()

    value = str(value).strip()

    if value == "":
        return set()

    return {
        int(page.strip())
        for page in value.split("|")
        if page.strip()
    }


def get_relevance(retrieved, gold_source, gold_pages):
    """
    1 = correct document + correct page
    0 = incorrect evidence
    """

    relevance = []

    for result in retrieved:

        correct_source = result["source"] == gold_source

        correct_page = result["page"] in gold_pages

        if correct_source and correct_page:
            relevance.append(1)

        else:
            relevance.append(0)

    return relevance


# ---------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------

def hit_at_k(relevance):
    """
    Returns 1 if at least one correct chunk was retrieved.
    """

    return 1 if any(relevance) else 0


def dcg(relevance):
    """
    Discounted Cumulative Gain.
    """

    result = 0.0

    for position, rel in enumerate(
        relevance,
        start=1
    ):

        result += rel / math.log2(
            position + 1
        )

    return result


def ndcg(relevance):
    """
    Normalised Discounted Cumulative Gain.
    """

    actual_dcg = dcg(
        relevance
    )

    ideal_relevance = sorted(
        relevance,
        reverse=True
    )

    ideal_dcg = dcg(
        ideal_relevance
    )

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


# ---------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------

def calculate_rouge(
    reference,
    generated
):

    scorer = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=True
    )

    scores = scorer.score(
        reference,
        generated
    )

    return scores[
        "rougeL"
    ].fmeasure


# ---------------------------------------------------------
# Check if system refused
# ---------------------------------------------------------

def is_unanswered(answer):

    answer = answer.lower()

    refusal_phrases = [
        "could not find enough information",
        "cannot find enough information",
        "not enough information",
        "not provided in the supplied documents",
        "not available in the supplied documents",
        "cannot be determined from the supplied documents"
    ]

    return any(
        phrase in answer
        for phrase in refusal_phrases
    )


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

def evaluate():

    questions = pd.read_csv(
        EVALUATION_FILE
    )

    results = []

    print(
        "\n========================================"
    )

    print(
        " Health Cover RAG Preliminary Evaluation"
    )

    print(
        "========================================\n"
    )

    # Run the same evaluation for each retriever
    for method in METHODS:

        if method == "dense":

            method_name = (
                "Dense FAISS"
            )

        else:

            method_name = (
                "BM25"
            )

        print(
            "\n========================================"
        )

        print(
            f" Evaluating {method_name}"
        )

        print(
            "========================================"
        )

        for cutoff in CUTOFFS:

            print(
                f"\nRunning evaluation with "
                f"Top-{cutoff} chunks"
            )

            print(
                "----------------------------------------"
            )

            for _, row in questions.iterrows():

                question_id = row[
                    "id"
                ]

                question_type = row[
                    "type"
                ]

                question = row[
                    "question"
                ]

                reference_answer = row[
                    "reference_answer"
                ]

                print(
                    f"{method_name} | "
                    f"{question_id}: "
                    f"{question}"
                )


                # --------------------------------------
                # Retrieval
                # --------------------------------------

                retrieved = retrieve(
                    question,
                    top_k=5,
                    method=method
                )

                selected_chunks = retrieved[
                    :cutoff
                ]


                # --------------------------------------
                # Generate answer
                # --------------------------------------

                generated_answer = (
                    generate_answer(
                        question,
                        selected_chunks
                    )
                )


                # --------------------------------------
                # Gold page data
                # --------------------------------------

                gold_pages = parse_gold_pages(
                    row["gold_pages"]
                )


                # --------------------------------------
                # Retrieval scores
                # --------------------------------------

                if question_type != "out_of_kb":

                    relevance = get_relevance(
                        selected_chunks,
                        row["gold_source"],
                        gold_pages
                    )

                    hit_score = hit_at_k(
                        relevance
                    )

                    ndcg_score = ndcg(
                        relevance
                    )

                else:

                    hit_score = None

                    ndcg_score = None


                # --------------------------------------
                # ROUGE
                # --------------------------------------

                rouge_l = calculate_rouge(
                    reference_answer,
                    generated_answer
                )


                # --------------------------------------
                # Was question unanswered?
                # --------------------------------------

                unanswered = is_unanswered(
                    generated_answer
                )


                # --------------------------------------
                # Save source information
                # --------------------------------------

                source_string = " | ".join(

                    f"{item['source']} "
                    f"p.{item['page']}"

                    for item in selected_chunks
                )


                results.append(
                    {
                        "approach":
                            method_name,

                        "id":
                            question_id,

                        "type":
                            question_type,

                        "cutoff":
                            cutoff,

                        "question":
                            question,

                        "reference_answer":
                            reference_answer,

                        "generated_answer":
                            generated_answer,

                        "hit":
                            hit_score,

                        "ndcg":
                            ndcg_score,

                        "rouge_l":
                            rouge_l,

                        "unanswered":
                            unanswered,

                        "retrieved_sources":
                            source_string
                    }
                )


    # --------------------------------------------------
    # Convert to dataframe
    # --------------------------------------------------

    results_df = pd.DataFrame(
        results
    )


    # --------------------------------------------------
    # BERTScore
    # --------------------------------------------------

    print(
        "\nCalculating BERTScore..."
    )

    print(
        "The first run may download "
        "an evaluation model.\n"
    )

    precision, recall, f1 = bert_score(

        results_df[
            "generated_answer"
        ].tolist(),

        results_df[
            "reference_answer"
        ].tolist(),

        lang="en",

        verbose=True
    )

    results_df[
        "bertscore"
    ] = f1.cpu().numpy()


    # --------------------------------------------------
    # Save detailed output
    # --------------------------------------------------

    results_df.to_csv(
        "evaluation_detailed_results.csv",
        index=False
    )


    # --------------------------------------------------
    # Create summary
    # --------------------------------------------------

    summary_rows = []

    for method in METHODS:

        if method == "dense":

            method_name = (
                "Dense FAISS"
            )

        else:

            method_name = (
                "BM25"
            )


        for cutoff in CUTOFFS:

            for question_type in [
                "known",
                "inferred",
                "out_of_kb"
            ]:

                subset = results_df[
                    (
                        results_df[
                            "approach"
                        ]
                        == method_name
                    )
                    &
                    (
                        results_df[
                            "cutoff"
                        ]
                        == cutoff
                    )
                    &
                    (
                        results_df[
                            "type"
                        ]
                        == question_type
                    )
                ]


                if question_type == "out_of_kb":

                    summary_rows.append(
                        {
                            "Approach":
                                method_name,

                            "Question Type":
                                "Out of KB",

                            "Top-k":
                                cutoff,

                            "Hit@k":
                                "-",

                            "NDCG@k":
                                "-",

                            "BERTScore":
                                subset[
                                    "bertscore"
                                ].mean(),

                            "ROUGE-L":
                                subset[
                                    "rouge_l"
                                ].mean(),

                            "% Unanswered":
                                subset[
                                    "unanswered"
                                ].mean()
                                * 100
                        }
                    )


                else:

                    summary_rows.append(
                        {
                            "Approach":
                                method_name,

                            "Question Type":
                                question_type.capitalize(),

                            "Top-k":
                                cutoff,

                            "Hit@k":
                                subset[
                                    "hit"
                                ].mean(),

                            "NDCG@k":
                                subset[
                                    "ndcg"
                                ].mean(),

                            "BERTScore":
                                subset[
                                    "bertscore"
                                ].mean(),

                            "ROUGE-L":
                                subset[
                                    "rouge_l"
                                ].mean(),

                            "% Unanswered":
                                subset[
                                    "unanswered"
                                ].mean()
                                * 100
                        }
                    )


    summary = pd.DataFrame(
        summary_rows
    )


    summary.to_csv(
        "evaluation_summary.csv",
        index=False
    )


    # --------------------------------------------------
    # Pretty printing
    # --------------------------------------------------

    display = summary.copy()

    numeric_columns = [
        "BERTScore",
        "ROUGE-L"
    ]


    for column in numeric_columns:

        display[column] = (
            display[column]
            .apply(
                lambda x:
                f"{x:.4f}"
            )
        )


    display[
        "% Unanswered"
    ] = display[
        "% Unanswered"
    ].apply(
        lambda x:
        f"{x:.1f}%"
    )


    for column in [
        "Hit@k",
        "NDCG@k"
    ]:

        display[column] = (
            display[column]
            .apply(
                lambda x:
                x
                if x == "-"
                else f"{float(x):.4f}"
            )
        )


    print(
        "\n========================================"
    )

    print(
        "       PRELIMINARY RESULTS"
    )

    print(
        "========================================\n"
    )


    print(
        tabulate(
            display,
            headers="keys",
            tablefmt="github",
            showindex=False
        )
    )


    print(
        "\nFiles created:"
    )

    print(
        "- evaluation_detailed_results.csv"
    )

    print(
        "- evaluation_summary.csv"
    )


if __name__ == "__main__":

    evaluate()