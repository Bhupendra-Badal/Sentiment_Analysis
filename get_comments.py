import csv
import os
import joblib
import googleapiclient.discovery
from collections import Counter
import html

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

svm_model = joblib.load('model/svmmodel.sav')
logistic_model = joblib.load('model/logistic.sav')
naive_model = joblib.load('model/naive_bayes.sav')

# Load the saved TF-IDF Vectorizer
vect_word = joblib.load('model/tfidf_vectorizer.sav')

# Define a regex pattern to match URLs
url_pattern = re.compile(r'https?://(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)')

# Define a function to remove URLs
def remove_url(text):
    return url_pattern.sub(r'', text)

# Preprocess the input text
def preprocess_text(text):
    text = text.lower()
    text = remove_url(text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Replace 'YOUR_API_KEY_HERE' with your actual YouTube Data API key
API_KEY = 'AIzaSyD_hodyDC5UDFoJrlcFQs0EeLq2E93VR-M'

def get_comments(video_id):
    # Disable OAuthlib's HTTPS verification when running locally.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=API_KEY)

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100
    )

    comments = []

    while request:
        response = request.execute()
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comment = html.unescape(comment)  # Decode HTML entities
            comments.append([comment])
        
        if 'nextPageToken' in response:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=response['nextPageToken']
            )
        else:
            request = None

    return comments

def write_comments_to_csv(comments, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Comment', 'Sentiment'])
        writer.writerows(comments)

def predict_sentiment(comments, model):
    # Assuming the model expects a list of comment texts for prediction
    texts = [comment[0] for comment in comments]
    texts = [preprocess_text(text) for text in texts]
    input_vect = vect_word.transform(texts)
    sentiments = model.predict(input_vect)
    
    # Convert numpy.int64 to int for JSON serialization
    sentiments = [int(sentiment) for sentiment in sentiments]
    
    comments_with_sentiment = [(comments[i][0], sentiment) for i, sentiment in enumerate(sentiments)]
    
    # Return unsorted comments with sentiment for immediate display
    return comments_with_sentiment

def calculate_sentiment_percentage(comments_with_sentiment):
    sentiments = [comment[1] for comment in comments_with_sentiment]
    sentiment_counts = Counter(sentiments)

    total = sum(sentiment_counts.values())
    positive = sentiment_counts.get(2, 0)
    neutral = sentiment_counts.get(1, 0)
    negative = sentiment_counts.get(0, 0)

    if total == 0:
        return "No comments to analyze."

    # Convert numpy.int64 to int for JSON serialization
    positive_percentage = float((positive / total) * 100)
    neutral_percentage = float((neutral / total) * 100)
    negative_percentage = float((negative / total) * 100)

    return {
        "positive_percentage": positive_percentage,
        "neutral_percentage": neutral_percentage,
        "negative_percentage": negative_percentage
        
}

if __name__ == "__main__":
    video_url = input("Enter the YouTube video URL: ")

    # Extract the video ID from the URL
    if 'v=' in video_url:
        video_id = video_url.split('v=')[1]
        ampersand_position = video_id.find('&')
        if ampersand_position != -1:
            video_id = video_id[:ampersand_position]
    else:
        print("Invalid YouTube URL")
        exit()

    # Load the comments from the YouTube video
    comments = get_comments(video_id)
    
    model = joblib.load('logistic_regression_model.joblib')

    # Predict the sentiment of the comments and return unsorted
    comments_with_sentiment = predict_sentiment(comments, model = naive_model)

    # Write the comments with sentiment to a CSV file
    write_comments_to_csv(comments_with_sentiment, "comments_with_sentiment.csv")
    sentiment_percentage = calculate_sentiment_percentage(comments_with_sentiment)

    # Calculate and print the sentiment percentage
    sentiment_percentage = calculate_sentiment_percentage(comments_with_sentiment)
    
    response = {
        "sentiment_percentage": sentiment_percentage,
        "comments": comments_with_sentiment
    }
    
    print(response['sentiment_percentage'])
    
