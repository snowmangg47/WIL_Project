# Group 13
1) Bilal Mohammed Beg (s42153546)
2) Hajar Al Gheilani (s4204060)
3) Samuel Amrita (s4200649)
4) Axa Gemini Lakra (s4225641)
5) Jagiello (s4216579)



# Walert - A Conversational Agent

## Setup

Install the Python dependencies from the `quantitative_eval` directory:

```bash
cd quantitative_eval
python -m pip install -r requirements.txt
```

The NLG evaluation uses NLTK tokenization for BLEU. Download the required tokenizer data once:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

The end-to-end statistical evaluation is implemented in R and requires `dplyr`:

```bash
conda install -c conda-forge r-base r-dplyr
```

Run the statistical evaluation from the `quantitative_eval` directory:

```bash
(cd src/nlg && Rscript end2end_eval.R)
```

# Evaluation Results
NDCG for Known and Inferred Questions
![NDCG](Evaluation_results/1.png)

% of unanswered out-of-knowledge-base questions 
![unanswere](Evaluation_results/2.png)

BERTScore
![BERTScore](Evaluation_results/3.png)


ROUGE-1
![ROUGE](Evaluation_results/4.png)



# Citation
If you use or reference this work, please cite it as follows:
```
@inproceedings{10.1145/3627508.3638309,
author = {Pathiyan Cherumanal, Sachin and Tian, Lin and Abushaqra, Futoon M. and Magnoss\~{a}o de Paula, Angel Felipe and Ji, Kaixin and Ali, Halil and Hettiachchi, Danula and Trippas, Johanne R. and Scholer, Falk and Spina, Damiano},
title = {Walert: Putting Conversational Information Seeking Knowledge into Action by Building and Evaluating a Large Language Model-Powered Chatbot},
year = {2024},
isbn = {9798400704345},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3627508.3638309},
doi = {10.1145/3627508.3638309},
booktitle = {Proceedings of the 2024 Conference on Human Information Interaction and Retrieval},
pages = {401–405},
numpages = {5},
keywords = {conversational information seeking, large language models, retrieval-augmented generation},
location = {<conf-loc>, <city>Sheffield</city>, <country>United Kingdom</country>, </conf-loc>},
series = {CHIIR '24}
}
```
