# Movie Recommendation System with K-Nearest Neighbors (KNN)

This project implements, evaluates, and demonstrates a dual-paradigm recommendation system using **Content-Based Filtering** and **Collaborative Filtering** via **K-Nearest Neighbors (KNN)** models. 

It features automated synthetic data generation for maximum reproducibility and offline execution, detailed Exploratory Data Analysis (EDA), metric validation, and a consolidated API wrapper for generating recommendations.

---

## 📖 Table of Contents
1. [Project Architecture](#-project-architecture)
2. [Dataset & Preparation](#-dataset--preparation)
3. [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
4. [Recommendation Models](#-recommendation-models)
   - [Content-Based Filtering](#content-based-filtering)
   - [Collaborative Filtering](#collaborative-filtering)
5. [Model Evaluation](#-model-evaluation)
6. [How to Use the Recommender](#-how-to-use-the-recommender)

---

## 🏗 Project Architecture

The pipeline is structured into clean, modular steps:
[ Data Source ] ➔ [ Data Cleaning/Deduplication ] ➔ [ EDA & Visualizations ] │ ┌─────────────────────────────────────┴─────────────────────────────────────┐ ▼ ▼ [ Content-Based Filter ] [ Collaborative Filter ] - Extracted Multi-Label Genres - Structured User-Item Sparse Matrix - MultiLabelBinarizer Encoding - Filled Unobserved Metrics with Zeros - Cosine Similarity Nearest Neighbors - User-User KNN Model Mapping


---

## 📊 Dataset & Preparation

The project dynamically provisions the data footprint using a hybrid collection mechanism:
- **Primary Source:** Downloads and processes a subset of MovieLens (150 unique users, 60 unique movies, 4,844 ratings).
- **Secondary/Fallback Source:** In case of connection failure, it programmatically synthesizes a matching dataset using standard distribution seeds to preserve functionality.
- **Data Pipeline Operations:**
  - Drops duplicates and handles missing metrics.
  - Enforces correct rating thresholds ($[0.5, 5.0]$).
  - Filters out unmapped entries (e.g. movies with "(no genres listed)").

---

## 📈 Exploratory Data Analysis (EDA)

Detailed distributions are visualized through seaborn: 
1. **Movie Rating Distribution:** Analyzes rating density indicating user behavior skewness.
2. **Top 10 Most Rated Movies:** Highlights popular movies to contextualize cold start recommendations.

---

## 🧠 Recommendation Models

### Content-Based Filtering
* **Feature Vectorization:** Transforms unstructured string multi-label genres via `MultiLabelBinarizer` into high-dimensional binary representation vectors.
* **Similarity Metric:** Calculates cosine distance among genre configurations to retrieve similar items.

### Collaborative Filtering
* **User-Item Matrix:** Maps rating vectors for users across all movies using a pivoted interaction grid.
* **User-User Recommendations:** Computes nearest neighbor users based on historical similarity weights. Predicts rating preferences for unrated movies based on weighted neighboring inputs.

---

## 🧪 Model Evaluation

We assess predictive reliability utilizing an **80/20 train/test split** over the observed user-item dataset:
- **Mean Absolute Error (MAE):** Evaluates absolute prediction errors (achieving ~0.7382 on clean baseline subsets).
- **Root Mean Squared Error (RMSE):** Penalizes larger forecasting anomalies (achieving ~0.9420 on clean baseline subsets).

---

## 🚀 How to Use the Recommender

The complete interface is wrapped in a unified interface `movie_recommender`.

```python
# 1. Get recommendations by Content Similarity (Movie to Movie)
movie_recommender(method="content", movie_name="Movie 1 (1991)", n_recommendations=5)

# 2. Get recommendations by Collaborative Habits (User to Movie)
movie_recommender(method="collaborative", user_id=1, n_recommendations=5)
