# YouTube Comment Sentiment Analysis Dashboard

A Streamlit-based NLP dashboard that analyses the sentiment of YouTube video comments using a RoBERTa transformer model. The app fetches video metadata and comments through the YouTube Data API v3, classifies each comment as **Positive**, **Neutral**, or **Negative**, and presents the results through interactive visualisations.

This project is useful for understanding audience reactions, monitoring public feedback, and exploring how sentiment changes across comments on a YouTube video.

---

## Demo

Run the app locally and paste any public YouTube video URL with comments enabled.

```bash
python -m streamlit run app.py
```

The app will open in your browser, usually at:

```text
http://localhost:8501
```

---

## Tech Stack

- **Python 3.11 recommended**
- **Streamlit** for the dashboard
- **YouTube Data API v3** for fetching video metadata and comments
- **Hugging Face Transformers** for sentiment analysis
- **RoBERTa** sentiment model
- **PyTorch** for model inference
- **Pandas** for data processing
- **Plotly** for interactive charts
- **WordCloud** and **Matplotlib** for word cloud generation

---

## Features

- Fetches YouTube video metadata, including title, channel, thumbnail, views, likes, and comment count
- Collects up to 200 public top-level comments from a YouTube video
- Performs sentiment classification using a RoBERTa transformer model
- Classifies each comment as **Positive**, **Neutral**, or **Negative**
- Displays KPI cards for total comments, sentiment percentages, and overall mood
- Shows sentiment distribution using a donut chart
- Shows sentiment confidence distribution using a histogram
- Visualises comment likes against sentiment confidence
- Displays the top liked comments by sentiment
- Generates a word cloud from comment text
- Includes a sortable and filterable comment explorer
- Allows users to export analysed results as a CSV file

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
├── .gitignore              # Files and folders ignored by Git
└── README.md               # Project documentation
```

---

## How the Application Works

1. The user enters a YouTube video URL.
2. The application extracts the video ID from the URL.
3. The YouTube Data API retrieves video metadata and public comments.
4. Each comment is passed through a RoBERTa sentiment model.
5. The model returns sentiment probabilities for positive, neutral, and negative classes.
6. The dashboard visualises the results through KPI cards, charts, comment cards, and a word cloud.
7. The user can filter comments and download the analysed results as a CSV file.

---

## Important Python Version Note

This project is recommended to run with **Python 3.11**.

Python 3.13 may cause installation issues with packages such as `tokenizers`, which is required by Hugging Face Transformers. If you see errors related to `tokenizers`, `maturin`, `pyo3`, or wheel building, create a fresh Python 3.11 environment and reinstall the dependencies.

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/youtube-sentiment-dashboard.git
cd youtube-sentiment-dashboard
```

Replace `your-username` with your actual GitHub username after creating the repository.

---

## 2. Create a Python environment

You can use either **Conda** or **venv**.

### Option A: Using Conda Recommended

```bash
conda create -n youtube_sentiment python=3.11 -y
conda activate youtube_sentiment
```

Check that the correct Python version is active:

```bash
python --version
```

You should see something like:

```text
Python 3.11.x
```

### Option B: Using venv

macOS / Linux:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install dependencies

First upgrade pip:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

The first installation may take some time because PyTorch and transformer-related packages can be large.

---

## YouTube API Key Setup

This project requires a YouTube Data API v3 key.

1. Go to the Google Cloud Console.
2. Create a new project or select an existing one.
3. Open **APIs & Services**.
4. Go to **Library**.
5. Search for **YouTube Data API v3**.
6. Enable the API.
7. Go to **Credentials**.
8. Click **Create Credentials**.
9. Select **API Key**.
10. Copy the generated key.

Create a `.env` file from the example file:

```bash
cp .env.example .env
```

Add your API key inside `.env`:

```env
YOUTUBE_API_KEY=your_api_key_here
```

Do **not** commit your real `.env` file to GitHub.

---

## Example `.env.example`

Create a file named `.env.example` and add:

```env
YOUTUBE_API_KEY=your_api_key_here
```

This file can be safely uploaded to GitHub because it does not contain a real API key.

---

## Run the Application

Use this command:

```bash
python -m streamlit run app.py
```

Using `python -m streamlit` is recommended because it makes sure Streamlit runs from the same Python environment where the dependencies were installed.

---

## Usage

1. Start the Streamlit app.
2. Paste a public YouTube video URL in the sidebar.
3. Select how many comments you want to analyse.
4. Choose whether to sort comments by relevance or time.
5. Click **Analyse Comments**.
6. View the sentiment results, charts, word cloud, and comment explorer.
7. Download the filtered results as a CSV file if needed.

---

## Model Used

This project uses:

```text
cardiffnlp/twitter-roberta-base-sentiment-latest
```

This model is designed for sentiment analysis on social media text, making it suitable for YouTube comments, which often include informal language, short phrases, emojis, and slang.

The model predicts three sentiment classes:

- **Positive**
- **Neutral**
- **Negative**

---

## Output Columns

After analysis, the app creates a DataFrame with comment information and sentiment results.

Example columns include:

| Column | Description |
|---|---|
| `comment_id` | Unique comment ID from YouTube |
| `text` | Comment text |
| `author` | Comment author name |
| `like_count` | Number of likes on the comment |
| `reply_count` | Number of replies |
| `published_at` | Comment publication date |
| `label` | Predicted sentiment label |
| `confidence` | Confidence score for the predicted label |
| `positive_score` | Positive sentiment probability |
| `neutral_score` | Neutral sentiment probability |
| `negative_score` | Negative sentiment probability |

---

## Recommended `.gitignore`

Create a `.gitignore` file and add:

```gitignore
# Virtual environments
venv/
.env/

# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd

# Environment variables
.env
.streamlit/secrets.toml

# macOS
.DS_Store

# Jupyter Notebook checkpoints
.ipynb_checkpoints/

# Local output files
*.csv
```

Important: keep your real `.env` file private.

---

## Troubleshooting

### 1. ModuleNotFoundError: No module named `googleapiclient`

Install the YouTube API client:

```bash
python -m pip install google-api-python-client
```

If the error continues, make sure you are installing and running the app from the same environment:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

---

### 2. Error while building `tokenizers`

If you see an error related to `tokenizers`, `maturin`, `pyo3`, or Python 3.13, use Python 3.11 instead:

```bash
conda create -n youtube_sentiment python=3.11 -y
conda activate youtube_sentiment
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Then run:

```bash
python -m streamlit run app.py
```

---

### 3. `YOUTUBE_API_KEY not found in .env file`

Make sure you created a `.env` file in the project root folder:

```text
youtube_sentiment/.env
```

The file should contain:

```env
YOUTUBE_API_KEY=your_api_key_here
```

---

### 4. Comments are disabled

Some YouTube videos have comments disabled. In that case, the app cannot fetch comments for that video. Try another public video with comments enabled.

---

## Possible Improvements

- Compare sentiment across multiple YouTube videos
- Add topic modelling to identify common discussion themes
- Analyse reply thread sentiment
- Add scheduled monitoring for changes in audience sentiment
- Deploy the dashboard on Streamlit Community Cloud
- Add support for multilingual sentiment analysis
- Add sarcasm detection for social media comments
- Add authentication for private dashboards

---

## Project Summary for Resume

Built an interactive NLP dashboard that analyses YouTube comment sentiment using a RoBERTa transformer model. Integrated the YouTube Data API v3 to fetch video metadata and comments, processed comments through a Hugging Face model, and visualised audience reactions using Streamlit, Plotly, KPI cards, word clouds, and CSV export functionality.

---

## License

This project is for educational and portfolio purposes. Add a license file if you plan to make the repository open source.
