# VoxForge AI Review Analytics

VoxForge AI Review Analytics is an end-to-end customer review analytics platform built as part of my AI Engineering bootcamp.

The project analyses Amazon product reviews using NLP and machine learning to identify customer sentiment, discover product topics, and generate business recommendations through an interactive dashboard.

## Live Demo

Frontend

https://voxforge-ai-review-analytics-jujq.vercel.app/

Backend API

https://voxforge-ai-review-analytics-production.up.railway.app/

Swagger Documentation

https://voxforge-ai-review-analytics-production.up.railway.app/docs

---

## Features

- Customer review sentiment analysis using DistilBERT
- Topic discovery with TF-IDF, Truncated SVD, L2 normalisation and K-Means clustering
- Topic-level sentiment analysis
- Deterministic Business priority and severity scoring
- Recommendation generation
- Interactive analytics dashboard
- REST API built with FastAPI

---

## Tech Stack

### Backend

- Python
- FastAPI
- Pandas
- Scikit-learn
- Hugging Face Transformers

### Machine Learning

- DistilBERT
- TF-IDF
- Truncated SVD
- L2 normalisation
- K-Means

### Frontend

- React
- TypeScript
- Vite
- Recharts

### Deployment

- Railway
- Vercel

---

## Project Structure

```text
voxforge-ai-review-analytics
│
├── backend
│   ├── app
│   │   ├── repository.py         # Load analytics artifacts
│   │   ├── routes.py             # FastAPI endpoints
│   │   ├── schemas.py            # API response models
│   │   └── services.py           # Business services
│   │
│   └── data
│       ├── dashboard.json
│       ├── topics.json
│       ├── recommendations.json
│       ├── report.json
│       └── report.md
│
├── frontend
│   ├── src
│   │   ├── components
│   │   ├── pages
│   │   ├── services
│   │   └── types
│   └── public
│
├── src
│   ├── data
│   │   ├── load.py
│   │   ├── text_preprocessing.py
│   │   ├── merge.py
│   │   ├── validate.py
│   │   └── schema.py
│   │
│   ├── sentiment
│   │   ├── transformer.py
│   │   ├── inference.py
│   │   ├── benchmarking.py
│   │   ├── comparison.py
│   │   ├── error_analysis.py
│   │   └── model_selection.py
│   │
│   ├── clustering
│   │   ├── tfidf_kmeans.py
│   │   └── labeling.py
│   │
│   ├── insights
│   │   ├── repository.py
│   │   ├── summary.py
│   │   ├── recommendation.py
│   │   ├── report.py
│   │   └── exporter.py
│   │
│   └── observability
│
├── notebooks
│   ├── 01_data_profiling_eda.ipynb
│   ├── 02_data_cleaning_preprocessing.ipynb
│   ├── 03_tfidf_logistic_regression.ipynb
│   ├── 04_distilbert_sentiment.ipynb
│   ├── 05_deberta_v3_lora_sentiment.ipynb
│   ├── 06_sentiment_model_selection_and_inference.ipynb
│   └── 07_tfidf_kmeans_topics.ipynb
│
├── models
│   ├── sentiment
│   └── clustering
│
├── data
│   ├── raw
│   ├── interim
│   └── processed
│
├── docs
│   ├── decisions
│   ├── model_cards
│   ├── screenshots
│   └── PHASE_1_TO_PHASE_2.md
│
├── tests
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Analytics Pipeline

1. Load and clean customer reviews
2. Predict review sentiment using DistilBERT
3. Convert reviews into TF-IDF vectors
4. Reduce dimensionality with Truncated SVD
5. Cluster reviews into product topics using K-Means
6. Calculate topic metrics
7. Generate business recommendations
8. Export analytics artifacts
9. Serve data through FastAPI
10. Visualize results in the React dashboard

---

## Running Locally

Clone the repository

```bash
git clone <repository-url>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the backend

```bash
uvicorn app.main:app --reload
```

Run the frontend

```bash
npm install
npm run dev
```

---

## API Documentation

Swagger UI

https://voxforge-ai-review-analytics-production.up.railway.app/docs

---

## Presentation Link

https://docs.google.com/presentation/d/1smy5FnNEMF1i4N8TBWelScUtTus5cDTk545K64jshpQ/edit?usp=sharing

---

## Author

Karima Mzoughi
