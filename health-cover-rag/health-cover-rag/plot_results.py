import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# SETTINGS
# =========================================================

RESULTS_FILE = "evaluation_detailed_results.csv"
OUTPUT_FOLDER = "graphs"

# Create graphs folder if it does not exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# =========================================================
# LOAD RESULTS
# =========================================================

df = pd.read_csv(RESULTS_FILE)

print("Loaded evaluation results.")
print(f"Number of rows: {len(df)}")

print("\nColumns:")
print(df.columns.tolist())


# =========================================================
# CREATE SUMMARY
# =========================================================

summary = (
    df.groupby(["type", "cutoff"])
    .agg(
        {
            "hit": "mean",
            "ndcg": "mean",
            "bertscore": "mean",
            "rouge_l": "mean",
            "unanswered": "mean",
        }
    )
    .reset_index()
)


print("\nEvaluation Summary:")
print(summary)


# Retrieval cutoffs
cutoffs = [1, 3, 5]

x = np.arange(len(cutoffs))

width = 0.35


# =========================================================
# GRAPH 1
# HIT@K
# =========================================================

known_hit = []
inferred_hit = []

for k in cutoffs:

    known_value = summary[
        (summary["type"] == "known")
        & (summary["cutoff"] == k)
    ]["hit"].iloc[0]

    inferred_value = summary[
        (summary["type"] == "inferred")
        & (summary["cutoff"] == k)
    ]["hit"].iloc[0]

    known_hit.append(known_value)
    inferred_hit.append(inferred_value)


plt.figure(figsize=(9, 6))

known_bars = plt.bar(
    x - width / 2,
    known_hit,
    width,
    label="Known"
)

inferred_bars = plt.bar(
    x + width / 2,
    inferred_hit,
    width,
    label="Inferred"
)

plt.bar_label(
    known_bars,
    fmt="%.2f",
    padding=3
)

plt.bar_label(
    inferred_bars,
    fmt="%.2f",
    padding=3
)

plt.xticks(
    x,
    [f"k={k}" for k in cutoffs]
)

plt.xlabel("Retrieval Cutoff")
plt.ylabel("Hit@k")

plt.title(
    "Retrieval Hit@k by Question Type"
)

plt.ylim(0, 1)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "hit_at_k.png"
    ),
    dpi=300
)

plt.show()


# =========================================================
# GRAPH 2
# NDCG@K
# =========================================================

known_ndcg = []
inferred_ndcg = []

for k in cutoffs:

    known_value = summary[
        (summary["type"] == "known")
        & (summary["cutoff"] == k)
    ]["ndcg"].iloc[0]

    inferred_value = summary[
        (summary["type"] == "inferred")
        & (summary["cutoff"] == k)
    ]["ndcg"].iloc[0]

    known_ndcg.append(known_value)
    inferred_ndcg.append(inferred_value)


plt.figure(figsize=(9, 6))

known_bars = plt.bar(
    x - width / 2,
    known_ndcg,
    width,
    label="Known"
)

inferred_bars = plt.bar(
    x + width / 2,
    inferred_ndcg,
    width,
    label="Inferred"
)

plt.bar_label(
    known_bars,
    fmt="%.3f",
    padding=3
)

plt.bar_label(
    inferred_bars,
    fmt="%.3f",
    padding=3
)

plt.xticks(
    x,
    [f"k={k}" for k in cutoffs]
)

plt.xlabel("Retrieval Cutoff")
plt.ylabel("NDCG@k")

plt.title(
    "NDCG@k by Question Type"
)

plt.ylim(0, 1)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "ndcg_at_k.png"
    ),
    dpi=300
)

plt.show()


# =========================================================
# GRAPH 3
# % UNANSWERED
# =========================================================

known_unanswered = []
inferred_unanswered = []
outkb_unanswered = []

for k in cutoffs:

    known_value = (
        summary[
            (summary["type"] == "known")
            & (summary["cutoff"] == k)
        ]["unanswered"].iloc[0]
        * 100
    )

    inferred_value = (
        summary[
            (summary["type"] == "inferred")
            & (summary["cutoff"] == k)
        ]["unanswered"].iloc[0]
        * 100
    )

    outkb_value = (
        summary[
            (summary["type"] == "out_of_kb")
            & (summary["cutoff"] == k)
        ]["unanswered"].iloc[0]
        * 100
    )

    known_unanswered.append(known_value)
    inferred_unanswered.append(inferred_value)
    outkb_unanswered.append(outkb_value)


width3 = 0.25

plt.figure(figsize=(10, 6))

known_bars = plt.bar(
    x - width3,
    known_unanswered,
    width3,
    label="Known"
)

inferred_bars = plt.bar(
    x,
    inferred_unanswered,
    width3,
    label="Inferred"
)

outkb_bars = plt.bar(
    x + width3,
    outkb_unanswered,
    width3,
    label="Out of KB"
)

plt.bar_label(
    known_bars,
    fmt="%.0f%%",
    padding=3
)

plt.bar_label(
    inferred_bars,
    fmt="%.0f%%",
    padding=3
)

plt.bar_label(
    outkb_bars,
    fmt="%.0f%%",
    padding=3
)

plt.xticks(
    x,
    [f"k={k}" for k in cutoffs]
)

plt.xlabel("Retrieval Cutoff")
plt.ylabel("% Unanswered")

plt.title(
    "Unanswered Rate by Question Type"
)

plt.ylim(0, 110)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "unanswered_rate.png"
    ),
    dpi=300
)

plt.show()


# =========================================================
# GRAPH 4
# BERTSCORE
# =========================================================

known_bert = []
inferred_bert = []
outkb_bert = []

for k in cutoffs:

    known_bert.append(
        summary[
            (summary["type"] == "known")
            & (summary["cutoff"] == k)
        ]["bertscore"].iloc[0]
    )

    inferred_bert.append(
        summary[
            (summary["type"] == "inferred")
            & (summary["cutoff"] == k)
        ]["bertscore"].iloc[0]
    )

    outkb_bert.append(
        summary[
            (summary["type"] == "out_of_kb")
            & (summary["cutoff"] == k)
        ]["bertscore"].iloc[0]
    )


plt.figure(figsize=(10, 6))

known_bars = plt.bar(
    x - width3,
    known_bert,
    width3,
    label="Known"
)

inferred_bars = plt.bar(
    x,
    inferred_bert,
    width3,
    label="Inferred"
)

outkb_bars = plt.bar(
    x + width3,
    outkb_bert,
    width3,
    label="Out of KB"
)

plt.bar_label(
    known_bars,
    fmt="%.3f",
    padding=3
)

plt.bar_label(
    inferred_bars,
    fmt="%.3f",
    padding=3
)

plt.bar_label(
    outkb_bars,
    fmt="%.3f",
    padding=3
)

plt.xticks(
    x,
    [f"k={k}" for k in cutoffs]
)

plt.xlabel("Retrieval Cutoff")
plt.ylabel("BERTScore")

plt.title(
    "BERTScore by Question Type"
)

plt.ylim(0, 1.1)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "bertscore.png"
    ),
    dpi=300
)

plt.show()


# =========================================================
# GRAPH 5
# ROUGE-L
# =========================================================

known_rouge = []
inferred_rouge = []
outkb_rouge = []

for k in cutoffs:

    known_rouge.append(
        summary[
            (summary["type"] == "known")
            & (summary["cutoff"] == k)
        ]["rouge_l"].iloc[0]
    )

    inferred_rouge.append(
        summary[
            (summary["type"] == "inferred")
            & (summary["cutoff"] == k)
        ]["rouge_l"].iloc[0]
    )

    outkb_rouge.append(
        summary[
            (summary["type"] == "out_of_kb")
            & (summary["cutoff"] == k)
        ]["rouge_l"].iloc[0]
    )


plt.figure(figsize=(10, 6))

known_bars = plt.bar(
    x - width3,
    known_rouge,
    width3,
    label="Known"
)

inferred_bars = plt.bar(
    x,
    inferred_rouge,
    width3,
    label="Inferred"
)

outkb_bars = plt.bar(
    x + width3,
    outkb_rouge,
    width3,
    label="Out of KB"
)

plt.bar_label(
    known_bars,
    fmt="%.3f",
    padding=3
)

plt.bar_label(
    inferred_bars,
    fmt="%.3f",
    padding=3
)

plt.bar_label(
    outkb_bars,
    fmt="%.3f",
    padding=3
)

plt.xticks(
    x,
    [f"k={k}" for k in cutoffs]
)

plt.xlabel("Retrieval Cutoff")
plt.ylabel("ROUGE-L")

plt.title(
    "ROUGE-L by Question Type"
)

plt.ylim(0, 1.1)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "rouge_l.png"
    ),
    dpi=300
)

plt.show()


# =========================================================
# FINISHED
# =========================================================

print("\nGraphs generated successfully!")

print("\nSaved inside:")
print(OUTPUT_FOLDER)

print("\nFiles:")

print("1. hit_at_k.png")
print("2. ndcg_at_k.png")
print("3. unanswered_rate.png")
print("4. bertscore.png")
print("5. rouge_l.png")