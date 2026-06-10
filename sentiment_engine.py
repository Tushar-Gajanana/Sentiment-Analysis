"""
sentiment_engine.py
--------------------
Runs sentiment analysis using cardiffnlp/twitter-roberta-base-sentiment-latest.
This model is trained on social media text — ideal for YouTube comments.
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

LABEL_MAP = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive",
}

EMOJI_MAP = {
    "Positive": "😊",
    "Neutral":  "😐",
    "Negative": "😔",
}

COLOR_MAP = {
    "Positive": "#2ecc71",
    "Neutral":  "#f39c12",
    "Negative": "#e74c3c",
}


@st.cache_resource(show_spinner="Loading sentiment model…")
def load_model():
    """Load and cache the tokeniser + model (runs only once per session)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def analyse_text(text: str, tokenizer, model) -> dict:
    """Classify a single piece of text. Returns label + per-class scores."""
    text = " ".join(str(text).split()[:400])  # rough token limit guard
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1).squeeze()

    scores = {LABEL_MAP[f"LABEL_{i}"]: float(probs[i]) for i in range(len(probs))}
    top_label = max(scores, key=scores.get)

    return {
        "label":           top_label,
        "confidence":      scores[top_label],
        "positive_score":  scores["Positive"],
        "neutral_score":   scores["Neutral"],
        "negative_score":  scores["Negative"],
    }


def analyse_dataframe(df: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
    """
    Add sentiment columns to a DataFrame with a 'text' column.
    Adds: label, confidence, positive_score, neutral_score, negative_score
    """
    tokenizer, model = load_model()
    results = []

    for i, text in enumerate(df["text"]):
        results.append(analyse_text(str(text), tokenizer, model))
        if progress_callback:
            progress_callback(i + 1, len(df))

    return pd.concat(
        [df.reset_index(drop=True), pd.DataFrame(results).reset_index(drop=True)],
        axis=1,
    )


def get_summary_stats(df: pd.DataFrame) -> dict:
    counts = df["label"].value_counts()
    total  = len(df)
    return {
        "total":             total,
        "positive_count":    int(counts.get("Positive", 0)),
        "neutral_count":     int(counts.get("Neutral",  0)),
        "negative_count":    int(counts.get("Negative", 0)),
        "positive_pct":      round(counts.get("Positive", 0) / total * 100, 1),
        "neutral_pct":       round(counts.get("Neutral",  0) / total * 100, 1),
        "negative_pct":      round(counts.get("Negative", 0) / total * 100, 1),
        "avg_positive_score":round(df["positive_score"].mean(), 3),
        "avg_negative_score":round(df["negative_score"].mean(), 3),
        "overall_sentiment": df["label"].mode()[0],
    }
