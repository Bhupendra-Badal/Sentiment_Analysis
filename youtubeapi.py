import csv
from googleapiclient.discovery import build

# API_KEY = "AIzaSyDDb0pYx9EmAtxbh8kEO_1TM0DltnO-AMU"

def get_video_comments(video_id):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    comments = []

    nextPageToken = None
    while True:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100,
            pageToken=nextPageToken
        ).execute()

        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        if "nextPageToken" in response:
            nextPageToken = response["nextPageToken"]
        else:
            break

    return comments

def get_comments_from_videos(video_ids):
    all_comments = []
    for video_id in video_ids:
        comments = get_video_comments(video_id)
        all_comments.extend(comments)
    return all_comments

video_ids = ["uEz8ob2aXoo", "sElE_BfQ67s"]  

all_comments = get_comments_from_videos(video_ids)

with open("youtube_comments.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Comments"])
    for comment in all_comments:
        writer.writerow([comment])

print("Comments have been saved to 'youtube_comments.csv'")
