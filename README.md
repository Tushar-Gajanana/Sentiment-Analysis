# ▶️ YouTube Comment Sentiment Analysis Dashboard

Analyse the sentiment of any YouTube video's comments using a RoBERTa
transformer model fine-tuned on 124M social media posts.

**Stack:** Python · Streamlit · HuggingFace Transformers · YouTube Data API v3 · Plotly

---

## 📁 Project Structure

```
youtube_sentiment/
├── app.py                  # Main Streamlit dashboard
├── youtube_fetcher.py      # YouTube Data API v3 integration
├── sentiment_engine.py     # HuggingFace RoBERTa sentiment model
├── visualisations.py       # Plotly charts + WordCloud
├── requirements.txt        # All dependencies
├── .env.example            # API key template
└── README.md
```

---

## 🚀 Setup (Step by Step)

### 1. Open the folder in VS Code
File → Open Folder → select `youtube_sentiment`

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> First install is slow (~2 GB for PyTorch). Grab a coffee ☕

### 4. Get a YouTube API key
1. Go to https://console.cloud.google.com
2. Create a new project
3. Go to **APIs & Services → Library**
4. Search **"YouTube Data API v3"** → Enable
5. Go to **APIs & Services → Credentials**
6. Click **"+ Create Credentials → API Key"**
7. Copy the key

### 5. Set up your .env file
```bash
# Rename the template
cp .env.example .env
```
Edit `.env`:
```
YOUTUBE_API_KEY=AIza...your_key_here
```

### 6. Run the dashboard
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## 🎛️ Features

| Feature | Description |
|---|---|
| **Video card** | Shows thumbnail, title, channel, views, likes |
| **5 KPI cards** | Total, Positive %, Neutral %, Negative %, Overall mood |
| **Donut chart** | Sentiment distribution |
| **Time series** | Sentiment scores over time |
| **Histogram** | Confidence score distribution |
| **Scatter plot** | Comment likes vs sentiment confidence |
| **Top comments bar** | Most liked comments coloured by sentiment |
| **Word cloud** | Common words filtered by sentiment |
| **Comment explorer** | Sortable, filterable comment browser |
| **CSV export** | Download results |

---

## 🧠 How It Works

1. User pastes a YouTube video URL
2. Video ID is extracted from the URL
3. YouTube Data API fetches metadata + up to 200 comments
4. Each comment is tokenised and passed through RoBERTa
5. Model outputs probability scores for Positive / Neutral / Negative
6. Results are visualised across 6 different charts

---

## 📈 CV Tips

- Deploy free on **Streamlit Community Cloud** (streamlit.io/cloud)
- Mention: *"Built NLP dashboard analysing YouTube comment sentiment using RoBERTa transformer; fetched 200 comments per video via YouTube Data API v3 and visualised results across 6 interactive charts"*
- Highlight model choice justification: RoBERTa over VADER for better accuracy on informal, emoji-heavy social media text

---

## 🔧 Possible Extensions

- [ ] Compare sentiment across multiple videos side by side
- [ ] Detect sarcasm layer on top of base sentiment
- [ ] Topic modelling (LDA) to find what people are positive/negative about
- [ ] Reply thread sentiment analysis
- [ ] Scheduled monitoring — alert when sentiment drops suddenly
