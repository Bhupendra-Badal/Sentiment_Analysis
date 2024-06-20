from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import joblib
import os
from pathlib import Path
from get_comments import get_comments, predict_sentiment, write_comments_to_csv, calculate_sentiment_percentage
import asyncio
import html
import json

app = FastAPI()



# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the index.html file
@app.get("/", response_class=HTMLResponse)
async def read_index():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=html_path.read_text(), status_code=200)

# Endpoint to analyze YouTube comments
@app.post("/analyze")
async def analyze_comments(request: Request):
    # Load the saved SVM model
    svm_model = joblib.load('model/svmmodel.sav')
    logistic_model = joblib.load('model/logistic.sav')
    naive_model = joblib.load('model/naive_bayes.sav')
    data = await request.json()
    video_url = data.get("video_url")
    selected_model = data.get("selected_model")
    if selected_model == "svm":
        model = svm_model
    elif selected_model == "logistic":
        model = logistic_model
    elif selected_model == "naive":
        model = naive_model
    else:
        return JSONResponse(content={"error": "Invalid model selection"}, status_code=400)
    
    if not video_url:
        return JSONResponse(content={"error": "No URL provided"}, status_code=400)
    
    # Extract the video ID from the URL
    if 'v=' in video_url:
        video_id = video_url.split('v=')[1]
        ampersand_position = video_id.find('&')
        if ampersand_position != -1:
            video_id = video_id[:ampersand_position]
    else:
        return JSONResponse(content={"error": "Invalid YouTube URL"}, status_code=400)
    
    # Load the comments from the YouTube video
    comments = get_comments(video_id)
    
    # Decode HTML entities in comments
    comments = [[html.unescape(comment[0])] for comment in comments]
    
    # Predict the sentiment of the comments
    comments_with_sentiment = predict_sentiment(comments,model)    
    
    # Calculate the sentiment percentage
    sentiment_percentage = calculate_sentiment_percentage(comments_with_sentiment)

    new =  {
        "sentiment_percentage" : sentiment_percentage,
        "comments" : comments_with_sentiment
    }
    
    
    return JSONResponse(content=new)

async def sort_and_write_comments_to_csv(comments_with_sentiment, filename):
    # Sort comments by sentiment (positive, neutral, negative)
    comments_with_sentiment.sort(key=lambda x: x[1])
    
    # Write the comments with sentiment to a CSV file
    write_comments_to_csv(comments_with_sentiment, filename)

# Endpoint to download the CSV file
@app.get("/download")
async def download_csv():
    csv_file_path = "comments_with_sentiment.csv"
    if not os.path.isfile(csv_file_path):
        raise HTTPException(status_code=404, detail="CSV file not found")
    return FileResponse(path=csv_file_path, filename="comments_with_sentiment.csv", media_type='text/csv')

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
