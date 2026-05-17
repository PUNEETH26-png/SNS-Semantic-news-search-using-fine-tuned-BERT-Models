# Semantic News Search using Fine-Tuned Transformer Models

A transformer-based semantic news classification and retrieval system built using Hugging Face Transformers and the AG News dataset. This project fine-tunes multiple BERT-based architectures for multi-class news classification and leverages contextual embeddings for semantic search and Top-K article retrieval.

---

## Overview

This project explores the use of pre-trained transformer models for:

* News article classification
* Semantic representation learning
* Embedding-based document retrieval
* Vector similarity search

The system fine-tunes transformer encoders on the AG News dataset and then uses the learned representations to perform semantic-aware search over news articles using embedding similarity.

---

## Features

* Fine-tuning of multiple transformer architectures
* Text preprocessing and normalization pipeline
* Hyperparameter experimentation
* Semantic embedding generation
* mean pooled embedding extraction
* Top-K semantic article retrieval
* Classification performance evaluation
* Confusion matrix visualization
* Cosine similarity based semantic search
* Added Faiss based retrieval search
---

## Models Used

The following transformer models were fine-tuned and evaluated:

* `bert-base-uncased`
* `distilbert-base-uncased`

All models were implemented using the Hugging Face Transformers library.

---

## Dataset

Dataset used: `SetFit/ag_news`

The AG News dataset contains news articles categorized into four classes:

* World
* Sports
* Business
* Sci/Tech

Dataset source:
https://huggingface.co/datasets/SetFit/ag_news

---

## Project Workflow

### 1. Dataset Preparation

* Loaded AG News dataset using Hugging Face Datasets
* Reduced dataset size for efficient experimentation
* Created train, validation, and test splits

### 2. Text Preprocessing

The following preprocessing operations were applied:

* Lowercasing
* Special character and punctuation removal
* Tokenization
* Stopword removal

A cleaned version of the article text was stored as `clean_text`.

### 3. Transformer Fine-Tuning

Each transformer model was fine-tuned using:

* Hugging Face Trainer API
* AutoTokenizer
* AutoModelForSequenceClassification

Hyperparameters such as learning rate and batch size were varied and evaluated.

### 4. Embedding Generation

After fine-tuning:

* Classification heads were bypassed
* mean pooled the outputs of last hidden states which are used as article embeddings 
* Article embeddings were generated for semantic retrieval

### 5. Semantic Search

The semantic retrieval system:

* Encodes user queries into embeddings
* Computes similarity against article embeddings
* Returns Top-K semantically relevant articles

Similarity search was implemented using cosine similarity and vector-based retrieval techniques.

### 6. Evaluation

Classification models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score (macro and micro)
* Confusion matrix

Semantic search quality was manually analyzed using multiple query examples.

---

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Scikit-learn
* NumPy
* Pandas
* NLTK

---


Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Open the notebook and execute all cells sequentially:

```bash
jupyter notebook
```

or run the notebook directly in Google Colab. you may need to give access to your google drive to run it seamlessly 

Main notebook:

```text
Semantic_News_Search_BERT.ipynb
```

---

## Example Semantic Search

Example query:

```text
global economic slowdown and stock market decline
```

The system retrieves the most semantically similar news articles along with:

* Category
* Similarity score
* Article snippet

---

## Results

The project compares multiple transformer architectures in terms of:

* Classification performance
* Embedding quality
* Semantic retrieval capability
* Computational efficiency

---

## Repository Structure


```text
.
├── notebooks/
│   └── Semantic_News_Search_BERT.ipynb
│
├── transformer_report.pdf
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Future Improvements

Possible extensions include:

* Cross-encoder reranking
* Hybrid BM25 + transformer retrieval
* Deployment as a web application
* Vector database integration
* Advanced hyperparameter optimization

---

## Acknowledgements

* Hugging Face Transformers
* Hugging Face Datasets
* AG News Dataset
* SetFit
* PyTorch

---

