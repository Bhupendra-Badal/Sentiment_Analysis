from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import os
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Load the pre-trained sentiment analysis model
model = joblib.load("B:\Sentiment Analysis\Try\model\Trained_move.joblib")

# Define the request body model
class VideoInput(BaseModel):
    video_id: str

# Initialize the FastAPI app
app = FastAPI()

# Mount the static directory for CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

# API key for YouTube Data API v3
API_KEY = os.getenv("AIzaSyBqcn1pQ8-t-Hsx2kWjUraUNuI3WpATOcc")

def get_video_comments(video_id):
    # Build the YouTube API client
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    
    comments = []
    try:
        results = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText"
        ).execute()

        while results:
            for item in results['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textOriginal']
                comments.append(comment)
            
            if 'nextPageToken' in results:
                results = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    pageToken=results['nextPageToken'],
                    maxResults=100,
                    textFormat="plainText"
                ).execute()
            else:
                break
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching comments: {e}")

    return comments

@app.get("/", response_class=HTMLResponse)
async def get_form():
    try:
        with open("templates/index.html", "r", encoding="utf-8") as file:
            return HTMLResponse(content=file.read(), status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading HTML template: {e}")

@app.post("/predict-sentiment/")
async def predict_sentiment(input: VideoInput):
    comments = get_video_comments(input.video_id)
    if not comments:
        return {"message": "No comments found or unable to fetch comments."}
    
    positive_count = 0
    neutral_count = 0
    total_comments = len(comments)
    
    for comment in comments:
        try:
            sentiment = model.predict([comment])[0]
            if sentiment == 'positive':
                positive_count += 1
            elif sentiment == 'neutral':
                neutral_count += 1
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error predicting sentiment: {e}")

    positive_percentage = (positive_count / total_comments) * 100
    neutral_percentage = (neutral_count / total_comments) * 100
    negative_percentage = 100 - positive_percentage - neutral_percentage

    return {
        "positive_percentage": positive_percentage,
        "negative_percentage": negative_percentage,
        "neutral_percentage": neutral_percentage,
        "total_comments": total_comments
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
