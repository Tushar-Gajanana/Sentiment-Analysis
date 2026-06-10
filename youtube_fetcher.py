"""
youtube_fetcher.py
------------------
Handles all YouTube Data API v3 interactions.
Fetches video metadata and comments from any public YouTube video.
"""

import os
import re
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()


def get_youtube_client():
    """Initialise and return a YouTube API client."""
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY not found in .env file.")
    return build("youtube", "v3", developerKey=api_key)


def extract_video_id(url_or_id: str) -> str:
    """
    Extract the video ID from a YouTube URL or return as-is if already an ID.

    Supports formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://youtube.com/shorts/VIDEO_ID
    - VIDEO_ID (raw)
    """
    patterns = [
        r"(?:v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:shorts/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)

    # If it looks like a raw video ID (11 chars)
    if re.match(r"^[a-zA-Z0-9_-]{11}$", url_or_id.strip()):
        return url_or_id.strip()

    raise ValueError(f"Could not extract a valid video ID from: {url_or_id}")


def fetch_video_metadata(video_id: str) -> dict:
    """
    Fetch title, channel, view count, like count, and description for a video.

    Returns:
        dict with video metadata
    """
    youtube = get_youtube_client()
    response = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    ).execute()

    if not response["items"]:
        raise ValueError(f"No video found with ID: {video_id}")

    item = response["items"][0]
    snippet = item["snippet"]
    stats = item.get("statistics", {})

    return {
        "video_id": video_id,
        "title": snippet.get("title", "Unknown"),
        "channel": snippet.get("channelTitle", "Unknown"),
        "published_at": snippet.get("publishedAt", ""),
        "description": snippet.get("description", "")[:300],
        "view_count": int(stats.get("viewCount", 0)),
        "like_count": int(stats.get("likeCount", 0)),
        "comment_count": int(stats.get("commentCount", 0)),
        "thumbnail": snippet.get("thumbnails", {}).get("medium", {}).get("url", ""),
        "url": f"https://www.youtube.com/watch?v={video_id}",
    }


def fetch_comments(
    video_id: str,
    max_comments: int = 100,
    sort_by: str = "relevance",
) -> pd.DataFrame:
    """
    Fetch top-level comments from a YouTube video.

    Args:
        video_id: YouTube video ID
        max_comments: number of comments to fetch (max 200)
        sort_by: 'relevance' or 'time'

    Returns:
        DataFrame with columns: comment_id, text, author, like_count,
        reply_count, published_at
    """
    youtube = get_youtube_client()
    records = []
    next_page_token = None
    fetched = 0

    while fetched < max_comments:
        batch_size = min(100, max_comments - fetched)

        try:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                order=sort_by,
                maxResults=batch_size,
                pageToken=next_page_token,
                textFormat="plainText",
            ).execute()
        except HttpError as e:
            if "commentsDisabled" in str(e):
                raise ValueError("Comments are disabled for this video.")
            raise e

        for item in response.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            records.append({
                "comment_id": item["id"],
                "text": top.get("textDisplay", ""),
                "author": top.get("authorDisplayName", "Anonymous"),
                "like_count": int(top.get("likeCount", 0)),
                "reply_count": int(item["snippet"].get("totalReplyCount", 0)),
                "published_at": datetime.strptime(
                    top["publishedAt"], "%Y-%m-%dT%H:%M:%SZ"
                ),
            })

        fetched += len(response.get("items", []))
        next_page_token = response.get("nextPageToken")

        if not next_page_token:
            break

    return pd.DataFrame(records)
