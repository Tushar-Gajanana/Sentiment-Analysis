"""
app.py
------
YouTube Comment Sentiment Analysis Dashboard
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from youtube_fetcher import (
    extract_video_id,
    fetch_video_metadata,
    fetch_comments,
)
from sentiment_engine import (
    analyse_dataframe,
    get_summary_stats,
    EMOJI_MAP,
    COLOR_MAP,
)
from visualisations import (
    plot_donut,
    plot_over_time,
    plot_histogram,
    plot_likes_vs_sentiment,
    plot_top_liked,
    generate_wordcloud,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube Sentiment Analyser",
    page_icon="▶️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    background-color: #0d0d1a;
    color: #e0e0e0;
    font-family: 'Syne', sans-serif;
}
.main-title {
    font-size: 2.6rem; font-weight: 800;
    background: linear-gradient(135deg, #ff4e45, #ff9a3c, #ffd166);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.1rem;
}
.subtitle {
    color: #666; font-size: 0.9rem;
    font-family: 'Space Mono', monospace; margin-bottom: 1.5rem;
}
.kpi {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-radius: 12px; padding: 1rem 1.2rem;
    border: 1px solid #2a2a4a; text-align: center;
}
.kpi-lbl { font-size: 8px; color: #666; text-transform: uppercase;
    letter-spacing: 0.1em; font-family: 'Space Mono', monospace; }
.kpi-val { font-size: 2rem; font-weight: 800; margin: 3px 0; }
.kpi-sub { font-size: 0.8rem; color: #888; }
.video-card {
    background: #1a1a2e; border-radius: 12px;
    padding: 1rem 1.2rem; border: 1px solid #2a2a4a;
    margin-bottom: 1rem; display: flex; gap: 1rem; align-items: flex-start;
}
.video-title { font-size: 1rem; font-weight: 700; margin-bottom: 4px; }
.video-meta { font-size: 0.78rem; color: #888; font-family: 'Space Mono', monospace; line-height: 1.8; }
.comment-card {
    background: #1a1a2e; border-radius: 8px;
    border-left: 3px solid; padding: 8px 12px; margin-bottom: 7px;
}
.comment-text { font-size: 0.88rem; margin-bottom: 4px; }
.comment-meta { font-size: 0.75rem; color: #666; font-family: 'Space Mono', monospace; }
.badge {
    display: inline-block; padding: 2px 9px; border-radius: 12px;
    font-size: 0.72rem; font-weight: 700; margin-right: 5px;
}
section[data-testid="stSidebar"] {
    background-color: #0a0a18; border-right: 1px solid #1e1e3a;
}
.stButton > button {
    background: linear-gradient(135deg, #ff4e45, #ff9a3c);
    color: white; border: none; border-radius: 8px;
    font-family: 'Syne', sans-serif; font-weight: 700;
    padding: 0.5rem 1.5rem; width: 100%;
}
.stButton > button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">▶️ YouTube Sentiment Analyser</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Powered by RoBERTa · YouTube Data API v3 · NLP Portfolio Project</div>',
    unsafe_allow_html=True,
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    video_url = st.text_input(
        "YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste any public YouTube video URL",
    )

    num_comments = st.slider("Comments to analyse", 20, 200, 100, step=20)

    sort_by = st.selectbox("Sort comments by", ["relevance", "time"], index=0)

    st.markdown("---")
    run_btn = st.button("🚀 Analyse Comments", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.73rem; color:#555; font-family: Space Mono, monospace; line-height:1.7;'>
    Model:<br>twitter-roberta-base<br>-sentiment-latest<br><br>
    Trained on 124M tweets.<br>
    3-class: Pos / Neu / Neg
    </div>
    """, unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key in ["df", "stats", "metadata"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Analysis ──────────────────────────────────────────────────────────────────
if run_btn:
    if not video_url.strip():
        st.error("Please paste a YouTube video URL.")
    else:
        with st.status("Fetching YouTube data…", expanded=True) as status:
            try:
                st.write("🔍 Extracting video ID…")
                video_id = extract_video_id(video_url.strip())

                st.write("📹 Fetching video metadata…")
                metadata = fetch_video_metadata(video_id)
                st.write(f"✅ Found: **{metadata['title']}**")

                st.write(f"💬 Fetching up to {num_comments} comments…")
                raw_df = fetch_comments(video_id, max_comments=num_comments, sort_by=sort_by)
                st.write(f"✅ Fetched {len(raw_df)} comments")

                st.write("🤖 Running sentiment analysis…")
                progress_bar = st.progress(0)

                def on_progress(cur, total):
                    progress_bar.progress(cur / total)

                df = analyse_dataframe(raw_df, progress_callback=on_progress)
                progress_bar.progress(1.0)

                stats = get_summary_stats(df)
                st.session_state.df       = df
                st.session_state.stats    = stats
                st.session_state.metadata = metadata
                status.update(label="✅ Analysis complete!", state="complete")

            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                st.info("Check that your YOUTUBE_API_KEY in .env is valid and YouTube Data API v3 is enabled.")

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.df is not None:
    df       = st.session_state.df
    stats    = st.session_state.stats
    meta     = st.session_state.metadata

    # ── Video info card ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="video-card">
        <img src="{meta['thumbnail']}" style="border-radius:8px; width:140px; flex-shrink:0;">
        <div>
            <div class="video-title">{meta['title']}</div>
            <div class="video-meta">
                📺 {meta['channel']}<br>
                👁 {meta['view_count']:,} views &nbsp;·&nbsp;
                👍 {meta['like_count']:,} likes &nbsp;·&nbsp;
                💬 {meta['comment_count']:,} comments<br>
                🔗 <a href="{meta['url']}" target="_blank" style="color:#ff9a3c;">Open on YouTube →</a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row ────────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("Analysed",  str(stats["total"]),          "comments",        "#7c83fd"),
        ("Positive",  f"{stats['positive_pct']}%",  f"{stats['positive_count']} 😊", COLOR_MAP["Positive"]),
        ("Neutral",   f"{stats['neutral_pct']}%",   f"{stats['neutral_count']} 😐",  COLOR_MAP["Neutral"]),
        ("Negative",  f"{stats['negative_pct']}%",  f"{stats['negative_count']} 😔", COLOR_MAP["Negative"]),
        ("Overall",   EMOJI_MAP[stats["overall_sentiment"]], stats["overall_sentiment"], COLOR_MAP[stats["overall_sentiment"]]),
    ]
    for col, (lbl, val, sub, color) in zip([c1,c2,c3,c4,c5], kpis):
        with col:
            st.markdown(f"""
            <div class="kpi">
                <div class="kpi-lbl">{lbl}</div>
                <div class="kpi-val" style="color:{color}">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row 1 ──────────────────────────────────────────────────────────
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.plotly_chart(plot_donut(stats), use_container_width=True)
    with col_b:
        if df["published_at"].nunique() > 1:
            st.plotly_chart(plot_over_time(df), use_container_width=True)
        else:
            st.plotly_chart(plot_histogram(df), use_container_width=True)

    # ── Charts row 2 ──────────────────────────────────────────────────────────
    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(plot_histogram(df), use_container_width=True)
    with col_d:
        st.plotly_chart(plot_likes_vs_sentiment(df), use_container_width=True)

    # ── Top liked comments chart ───────────────────────────────────────────────
    st.plotly_chart(plot_top_liked(df, n=10), use_container_width=True)

    # ── Word cloud ─────────────────────────────────────────────────────────────
    st.markdown("### 🔤 Word Cloud")
    wc_filter = st.radio(
        "Filter by sentiment",
        ["All", "Positive", "Neutral", "Negative"],
        horizontal=True,
    )
    try:
        st.pyplot(generate_wordcloud(df, wc_filter), use_container_width=True)
    except Exception as e:
        st.warning(f"Word cloud error: {e}")

    # ── Comment explorer ───────────────────────────────────────────────────────
    st.markdown("### 💬 Comment Explorer")

    col_f, col_g = st.columns([2, 1])
    with col_f:
        sentiment_filter = st.multiselect(
            "Filter by sentiment",
            ["Positive", "Neutral", "Negative"],
            default=["Positive", "Neutral", "Negative"],
        )
    with col_g:
        sort_col = st.selectbox("Sort by", ["like_count", "published_at", "confidence"])

    filtered = (
        df[df["label"].isin(sentiment_filter)]
        .sort_values(sort_col, ascending=False)
        .head(30)
    )

    for _, row in filtered.iterrows():
        border = COLOR_MAP.get(row["label"], "#555")
        emoji  = EMOJI_MAP.get(row["label"], "")
        text   = str(row["text"])[:200] + ("…" if len(str(row["text"])) > 200 else "")
        date   = pd.to_datetime(row["published_at"]).strftime("%d %b %Y")

        st.markdown(f"""
        <div class="comment-card" style="border-left-color:{border}">
            <div class="comment-text">{text}</div>
            <div class="comment-meta">
                <span class="badge" style="background:{border}22; color:{border}">
                    {emoji} {row['label']} ({row['confidence']:.0%})
                </span>
                👤 {row['author']} &nbsp;·&nbsp;
                👍 {int(row['like_count'])} likes &nbsp;·&nbsp;
                🕐 {date}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Export ─────────────────────────────────────────────────────────────────
    st.markdown("### 💾 Export")
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download CSV",
        data=csv,
        file_name=f"youtube_sentiment_{meta['video_id']}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

else:
    # ── Empty state ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding:5rem 2rem; color:#444;">
        <div style="font-size:4rem">▶️</div>
        <div style="font-size:1.2rem; font-weight:700; margin-top:1rem; color:#666;">
            Paste a YouTube URL to get started
        </div>
        <div style="font-size:0.9rem; margin-top:0.5rem; color:#444;">
            Works with any public video that has comments enabled
        </div>
    </div>
    """, unsafe_allow_html=True)
