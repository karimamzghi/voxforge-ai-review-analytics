# Phase 1 completion and Phase 2 start

## Execute in this order

1. Run the existing TF-IDF, DistilBERT and DeBERTa notebooks until each available model and validation prediction file is saved.
2. Run `notebooks/sentiments/06_sentiment_model_selection_and_inference.ipynb`.
3. Review `results/sentiment_model_comparison.md` and `results/error_analysis/`.
4. Confirm `data/processed/reviews_with_sentiment.csv` exists.
5. Run `notebooks/clustering/07_tfidf_kmeans_topics.ipynb`.
6. Inspect top terms and representative reviews, then replace placeholder topic names.

## Ablations

Ablations are intentionally not automated into a large experiment framework. Re-run a controlled transformer configuration and record it with `record_ablation()` from `src/sentiment/ablation.py`. Recommended minimum:

- class weights on versus off;
- max length 128 versus 256.

Use the same split, seed and epochs for a valid comparison.
