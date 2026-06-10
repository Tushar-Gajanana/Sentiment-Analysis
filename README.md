# YouTube Comment Sentiment Analysis Dashboard

A Streamlit-based NLP dashboard that analyses the sentiment of YouTube video comments using a RoBERTa transformer model. The app fetches video metadata and comments through the YouTube Data API v3, classifies each comment as positive, neutral, or negative, and presents the results through interactive visualisations.

This project is useful for understanding audience reactions, monitoring public feedback, and exploring how sentiment changes across comments on a YouTube video.

---

## Tech Stack

- **Python**
- **Streamlit**
- **Hugging Face Transformers**
- **RoBERTa sentiment model**
- **YouTube Data API v3**
- **Plotly**
- **WordCloud**
- **Pandas**

---

## Features

- Fetches YouTube video metadata, including title, channel, thumbnail, views, and likes
- Collects up to 200 public comments from a YouTube video
- Performs sentiment classification using a RoBERTa transformer model
- Displays positive, neutral, and negative sentiment distribution
- Shows KPI cards for total comments, sentiment percentages, and overall mood
- Provides interactive visualisations using Plotly
- Generates a word cloud based on comment text
- Includes a sortable and filterable comment explorer
- Allows users to export the analysed results as a CSV file

---

## Project Structure

```text
youtube_sentiment/
├── app.py                  # Main Streamlit dashboard
├── youtube_fetcher.py      # YouTube Data API v3 integration
├── sentiment_engine.py     # RoBERTa sentiment analysis logic
├── visualisations.py       # Plotly charts and word cloud generation
├── requirements.txt        # Project dependencies
├── .env.example            # Example environment variable file
└── README.md               # Project documentation
```

---

## How the Application Works

1. The user enters a YouTube video URL.
2. The application extracts the video ID from the URL.
3. The YouTube Data API retrieves video metadata and public comments.
4. Each comment is passed through a RoBERTa sentiment model.
5. The model returns sentiment probabilities for positive, neutral, and negative classes.
6. The dashboard visualises the results through KPI cards, charts, tables, and a word cloud.

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/youtube-sentiment-dashboard.git
cd youtube-sentiment-dashboard
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate the environment:

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The first installation may take some time because the transformer and PyTorch-related packages can be large.

---

## YouTube API Key Setup

This project requires a YouTube Data API v3 key.

1. Go to the Google Cloud Console.
2. Create a new project or select an existing one.
3. Open **APIs & Services**.
4. Enable **YouTube Data API v3**.
5. Go to **Credentials**.
6. Create a new API key.
7. Copy the generated key.

Create a `.env` file from the example file:

```bash
cp .env.example .env
```

Add your API key inside `.env`:

```env
YOUTUBE_API_KEY=your_api_key_here
```

Do not commit your real `.env` file to GitHub.

---

## Run the Application

```bash
streamlit run app.py
```

After running the command, open the local Streamlit URL in your browser, usually:

```text
http://localhost:8501
```

---

## Example Use Case

A user can paste a YouTube video URL into the dashboard to analyse how viewers reacted to the video. The dashboard then shows the overall sentiment distribution, the most liked comments, confidence scores, and common words appearing in the comment section.

---

## Environment Variables

Create a `.env` file with the following variable:

```env
YOUTUBE_API_KEY=your_api_key_here
```

The `.env.example` file should be committed to GitHub, but the real `.env` file should be ignored using `.gitignore`.

---

## Recommended `.gitignore`

```gitignore
venv/
__pycache__/
*.pyc
.env
.streamlit/secrets.toml
.DS_Store
```

---

## Possible Improvements

- Compare sentiment across multiple YouTube videos
- Add topic modelling to identify common discussion themes
- Analyse sentiment in reply threads
- Add scheduled monitoring for changes in audience sentiment
- Deploy the dashboard on Streamlit Community Cloud
- Add support for multilingual sentiment analysis

---

## Project Summary for Resume

Built an NLP dashboard that analyses YouTube comment sentiment using a RoBERTa transformer model. Integrated the YouTube Data API v3 to fetch video metadata and comments, processed comments through a Hugging Face sentiment pipeline, and visualised results through interactive Streamlit and Plotly components.

---

## License

This project is for educational and portfolio purposes. Add a license file if you plan to make the repository open source.
