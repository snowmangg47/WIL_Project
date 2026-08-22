# Walert - A Conversational Agent

Demo Video Link: https://bit.ly/chiir24walertdemovideo

## About Walert

Walert is a conversational agent designed to answer frequently asked questions (FAQs) regarding programs of study offered at the School of Computing Technologies, RMIT University. Our intent-based approach, deployed on Amazon Echo devices, was showcased as a live demo during RMIT University's Open Day in August 2023.

Note: This repository contains all utility code for 'Behind The Scenes' of Walert.

## Usage

Run the retrieval reproduction from the `quantitative_eval` directory:

```bash
bash src/retrieval/main.sh
```

This prepares the evaluation data, rebuilds the BM25 and dense retrieval indexes, produces retrieval runs, and evaluates the known and inferred question sets.

The individual scripts can also be run separately:

- `data.py` prepares our retrieval inputs.
- `index-bm25.sh` builds our BM25 index.
- `encode.sh` encodes our passage collection for dense retrieval.
- `index.sh` builds our dense FAISS index.
- `search.py` runs our BM25 or dense retrieval.
- `eval.py` evaluates retrieval runs.

Reproduced indexes and runs are written under `target/repro/`. The released author outputs elsewhere under `target/` are kept as reference artifacts.

`RAG_SYSTEM.py` provides the practical local text-based RAG interface.

`RAG_Voice_Demo.py` is the original voice demonstration pipeline. It uses microphone input, Whisper speech recognition, dense retrieval, an external Falcon API endpoint, and text-to-speech. It is preserved as a legacy demo and is not required for the retrieval reproduction above.
