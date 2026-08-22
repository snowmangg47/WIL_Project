# Walert - A Conversational Agent
Demo Video Link: https://bit.ly/chiir24walertdemovideo

## About Walert
Walert is a conversational agent designed to answer frequently asked questions (FAQs) regarding programs of study offered at the School of Computing Technologies, RMIT University. Our intent-based approach, deployed on Amazon Echo devices, was showcased as a live demo during RMIT University's Open Day in August 2023.


Note: This repository contains all utility code for 'Behind The Scenes' of Walert.

## intent-based

Contains the Alexa-based intent implementation of Walert, including the interaction models for the supported locales and the Lambda handlers that map intents to responses.

## nlg

Contains the response generation and evaluation code used for Walert's NLG experiments, including Falcon generation, ROUGE, BLEU, BERTScore, and the end-to-end statistical evaluation.

## retrieval

Contains the BM25 and dense retrieval reproduction pipeline, the local text-based RAG system, and the original voice demonstration pipeline.

See `retrieval/README.md` for retrieval usage and reproduction instructions.
