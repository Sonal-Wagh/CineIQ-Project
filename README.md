# CineIQ: Explainable & Sentiment-Aware Movie Recommendation Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cineiq-project-62i58kefughcosj6xxvatz.streamlit.app/#cine-iq)

## Problem Statement
Content discovery on modern streaming platforms is often opaque, biased toward promoted titles, and traps users in repetitive recommendation loops. **CineIQ** solves this by providing an open, explainable movie recommendation engine. It combines multiple machine learning strategies to deliver personalized, interpretable suggestions that evolve with your true taste over time.

---

## Deliverables & Key Features

### 1. Hybrid Recommendation Engine
CineIQ balances your historical preferences with the actual content of the movies using a weighted ensemble:
* **Content-Based Filtering:** Uses a TF-IDF vectorizer and cosine similarity on movie metadata (genres, keywords, cast, crew, and plot overview).
* **Collaborative Filtering:** Uncovers latent user preferences through SVD-based matrix factorization applied to user rating histories.

### 2. Sentiment-Aware Re-Ranker
To ensure recommendations align with real audience reception, CineIQ re-ranks candidates using advanced Natural Language Processing (NLP) on movie reviews:
* Utilizes **VADER** for baseline sentiment scoring.
* Leverages a fine-tuned **HuggingFace DistilBERT** model for deep contextual sentiment analysis.

### 3. Explainability Layer (XAI)
No more "black box" algorithms. Every recommendation surfaces a human-readable reason for why it was chosen:
* **Rule-Based Templates:** Provides clear logic (e.g., *"Recommended because it is directed by Christopher Nolan, and is also a Crime/Action film."*).
* **LIME (Local Interpretable Model-agnostic Explanations):** Extracts and highlights the specific textual signals or keywords that influenced the model's decision.

### 4. User Taste Dashboard
A highly interactive frontend built with Streamlit and Plotly that visualizes your cinematic profile, including:
* Genre radar charts.
* Decade preferences.
* Director and actor affinities derived from historical rating data.

---

## Tech Stack

* **Machine Learning:** Python, scikit-learn, Surprise (SVD), Pandas, NumPy
* **NLP & Explainability:** VADER, HuggingFace DistilBERT, LIME
* **Model Serving:** FastAPI (`/recommend` and `/similar` endpoints), Uvicorn
* **Frontend & Visualization:** Streamlit, Plotly
* **Experiment Tracking:** MLflow

---

## Datasets

The engine is trained on industry-standard, large-scale datasets:
1. **MovieLens 25M:** [grouplens.org/datasets/movielens/25m](https://grouplens.org/datasets/movielens/25m/)
2. **TMDB Metadata (Kaggle):** Cast, genres, and keywords for over 45K movies.
3. **IMDB 50K Reviews (Kaggle):** Utilized for training the sentiment analysis models.

