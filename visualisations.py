"""
visualisations.py
-----------------
All Plotly charts and WordCloud figures for the YouTube sentiment dashboard.
"""

import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sentiment_engine import COLOR_MAP

POS  = COLOR_MAP["Positive"]
NEU  = COLOR_MAP["Neutral"]
NEG  = COLOR_MAP["Negative"]
BG   = "rgba(0,0,0,0)"

STOPWORDS = {
    "the","a","an","is","it","in","of","and","to","for","that","this","with",
    "are","was","be","have","on","at","by","from","or","but","not","they","he",
    "she","we","you","i","my","your","its","do","did","will","just","so","if",
    "as","up","out","about","what","who","how","all","one","when","can","video",
    "watch","youtube","like","comment","subscribe","channel","https","www","com",
    "really","very","much","more","also","get","been","has","had","him","her",
    "their","there","than","then","would","could","should","its","dont","doesnt",
}


def _base_layout(title: str) -> dict:
    return dict(
        title=title,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color="#e0e0e0"),
        margin=dict(t=50, b=30, l=20, r=20),
    )


# ── Donut ─────────────────────────────────────────────────────────────────────
def plot_donut(stats: dict) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=["Positive", "Neutral", "Negative"],
        values=[stats["positive_count"], stats["neutral_count"], stats["negative_count"]],
        hole=0.55,
        marker=dict(colors=[POS, NEU, NEG], line=dict(color="#1a1a2e", width=2)),
        textinfo="label+percent",
        textfont=dict(size=12),
    ))
    fig.update_layout(**_base_layout("Sentiment Distribution"))
    return fig


# ── Sentiment over time ───────────────────────────────────────────────────────
def plot_over_time(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    df["date"] = pd.to_datetime(df["published_at"]).dt.date
    daily = (
        df.groupby("date")
        .agg(positive=("positive_score","mean"),
             neutral =("neutral_score", "mean"),
             negative=("negative_score","mean"))
        .reset_index()
    )
    fig = go.Figure()
    for col, color, name in [("positive",POS,"Positive"),
                               ("neutral", NEU,"Neutral"),
                               ("negative",NEG,"Negative")]:
        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily[col], name=name,
            line=dict(color=color, width=2), mode="lines+markers",
            marker=dict(size=5),
        ))
    fig.update_layout(
        **_base_layout("Sentiment Over Time"),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.25),
        yaxis=dict(gridcolor="#2a2a4a"),
        xaxis=dict(gridcolor="#2a2a4a"),
    )
    return fig


# ── Confidence histogram ──────────────────────────────────────────────────────
def plot_histogram(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for label, color in COLOR_MAP.items():
        fig.add_trace(go.Histogram(
            x=df[df["label"] == label]["confidence"],
            name=label, marker_color=color, opacity=0.75, nbinsx=20,
        ))
    fig.update_layout(
        **_base_layout("Confidence Score Distribution"),
        barmode="overlay",
        xaxis_title="Confidence",
        yaxis_title="Count",
        yaxis=dict(gridcolor="#2a2a4a"),
        xaxis=dict(gridcolor="#2a2a4a"),
    )
    return fig


# ── Likes vs sentiment ────────────────────────────────────────────────────────
def plot_likes_vs_sentiment(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df, x="like_count", y="confidence",
        color="label", color_discrete_map=COLOR_MAP,
        hover_data=["author","text"],
        title="Comment Likes vs Sentiment Confidence",
        labels={"like_count":"Likes on Comment", "confidence":"Sentiment Confidence"},
    )
    fig.update_layout(
        **_base_layout("Comment Likes vs Sentiment Confidence"),
        yaxis=dict(gridcolor="#2a2a4a"),
        xaxis=dict(gridcolor="#2a2a4a"),
    )
    return fig


# ── Top comments bar chart ────────────────────────────────────────────────────
def plot_top_liked(df: pd.DataFrame, n: int = 10) -> go.Figure:
    top = df.nlargest(n, "like_count")[["text","like_count","label"]].copy()
    top["short_text"] = top["text"].str[:50] + "…"
    top["color"] = top["label"].map(COLOR_MAP)

    fig = go.Figure(go.Bar(
        x=top["like_count"],
        y=top["short_text"],
        orientation="h",
        marker_color=top["color"].tolist(),
        text=top["label"],
        textposition="outside",
    ))
    fig.update_layout(
        **_base_layout(f"Top {n} Most Liked Comments"),
        xaxis_title="Likes",
        yaxis=dict(autorange="reversed"),
        height=400,
    )
    return fig


# ── Word cloud ────────────────────────────────────────────────────────────────
def generate_wordcloud(df: pd.DataFrame, sentiment_filter: str = "All") -> plt.Figure:
    subset = df if sentiment_filter == "All" else df[df["label"] == sentiment_filter]
    combined = " ".join(subset["text"].dropna().astype(str))
    combined = re.sub(r"http\S+|www\S+", "", combined)
    combined = re.sub(r"[^a-zA-Z\s]", " ", combined)

    color_funcs = {
        "Positive": lambda *a, **k: POS,
        "Neutral":  lambda *a, **k: NEU,
        "Negative": lambda *a, **k: NEG,
        "All":      lambda *a, **k: "#7c83fd",
    }

    wc = WordCloud(
        width=900, height=400,
        background_color="#0d0d1a",
        stopwords=STOPWORDS,
        color_func=color_funcs.get(sentiment_filter, color_funcs["All"]),
        max_words=120,
        collocations=False,
    ).generate(combined)

    fig, ax = plt.subplots(figsize=(11, 5), facecolor="#0d0d1a")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig
