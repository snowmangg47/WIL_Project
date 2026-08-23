import pandas as pd

# Load the evaluation results

# Intent-Based results
intent_based = pd.read_csv(r"C:\Users\jxjh1\OneDrive\Documents\g\WIL_Project\quantitative_eval\target\summaries\walert_eval.csv")

# RAG results using BM25
bm25 = pd.read_csv(r"C:\Users\jxjh1\OneDrive\Documents\g\WIL_Project\quantitative_eval\target\summaries\falcon_bm25_eval.csv")

# RAG results using dense vector retrieval with FAISS
dense_faiss = pd.read_csv(r"C:\Users\jxjh1\OneDrive\Documents\g\WIL_Project\quantitative_eval\target\summaries\falcon_dense_eval.csv")


# Function for the intent-based approach to calculate the avg. evaluation score
def summarise_intent_based(df):

    summary = {
        "Approach": "Intent-Based (IB)",
        "Cutoff k": "-",

        "ROUGE-1": df["rouge_1_f1"].mean(),
        "ROUGE-2": df["rouge_2_f1"].mean(),
        "ROUGE-L": df["rouge_l_f1"].mean(),

        "BLEU": df["bleu_score"].mean(),

        "BERTScore": df["bert_score_f1"].mean()
    }
    return summary

# Function for the RAG approaches to calculate the avg. evaluation
# cutoff_k: the number of top retrieved passages provided when generating the final answer
# k = # where falcon receives the top 1, 3, 5 retrieved passage/s

def summarise_rag(df, approach_name, cutoff_k):
    summary = {
        "Approach": approach_name,
        "Cutoff k": cutoff_k,

    "ROUGE-1": df[f"rouge_1_f1_top{cutoff_k}"].mean(),
    "ROUGE-2": df[f"rouge_2_f1_top{cutoff_k}"].mean(),
    "ROUGE-L": df[f"rouge_l_f1_top{cutoff_k}"].mean(),

    "BLEU": df[f"bleu_score_top{cutoff_k}"].mean(),

    "BERTScore": df[f"bert_score_f1_top{cutoff_k}"].mean()

    }

    return summary

# Summarise the intent-based results
intent_summary = summarise_intent_based(intent_based)

# Summarise BM25 + Falcon results
# when k = 1
bm25_k1 = summarise_rag(
    bm25,
    "RAG (BM25 + Falcon)",
    1
)

# k = 3
bm25_k3 = summarise_rag(
    bm25,
    "RAG (BM25 + Falcon)",
    3
)

# k = 5
bm25_k5 = summarise_rag(
    bm25,
    "RAG (BM25 + Falcon)",
    5
)

# Summarise dense FAISS + Falcon results
# k = 1
dense_k1 = summarise_rag(
    dense_faiss,
    "RAG (Dense FAISS + Falcon)",
    1
)

# k = 3
dense_k3 = summarise_rag(
    dense_faiss,
    "RAG (Dense FAISS + Falcon)",
    3
)

# k = 5
dense_k5 = summarise_rag(
    dense_faiss,
    "RAG (Dense FAISS + Falcon)",
    5
)

# Results table
results = pd.DataFrame([
    intent_summary,
    bm25_k1,
    bm25_k3,
    bm25_k5,
    dense_k1,
    dense_k3,
    dense_k5,
])

# Round the scores to 4 decimal places
metric_columns = [
    "ROUGE-1",
    "ROUGE-2",
    "ROUGE-L",
    "BLEU",
    "BERTScore"
]

results[metric_columns] = results[metric_columns].round(4)

# Save table

results.to_csv(
    "evaluation_results.csv",
    index=False,
)