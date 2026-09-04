# Group 13
1) Bilal Mohammed Beg (s4153547)
2) Hajar Al Gheilani (s4204060)
3) Samuel Amrita (s4200649)
4) Axa Gemini Lakra (s4225641)
5) Jagiello (s4216579)



# Walert - A Conversational Agent

## First-Time Setup
The model we are running is located in the `quantitative_eval` directory and we first create the environment
```bash
conda create -n group13rag python=3.12.9 -y 
conda activate group13rag
```

For Apple Silicon Macs: 
```bash
python -m pip install mlx-lm==0.31.3

brew install libomp pcre

python -m pip install faiss-cpu==1.14.3
conda install -c conda-forge lxml=4.9.3 -y
conda install -c conda-forge lightgbm -y

# Note: the current local --chat implementation is designed for Apple Silicon because it uses MLX.
```

Windows
Run the following in PowerShell:
```bash
conda install -c conda-forge lxml=4.9.3 lightgbm faiss-cpu -y
```


Then we need to download the dependencies:
```bash
cd quantitative_eval # Put the quantitative eval filepath here c: 
python -m pip install -r requirements.txt
```

Install Java
Walert uses Pyserini/PyJNIus, which require a Java Development Kit (JDK).
```bash
conda install -c conda-forge openjdk=21.0.10 -y
```

Download the NLTK Tokenizer Data
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

## Quick Start
```bash
cd quantitative_eval # Put the quantitative eval filepath here c: 
conda activate group13rag

src/retrieval/main.sh
python src/retrieval/RAG_SYSTEM.py --chat
```


# Evaluation Results
Template here:
![Results](example.png)





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
