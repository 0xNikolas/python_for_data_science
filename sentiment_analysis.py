import tweepy
from textblob import TextBlob
import csv
from config import settings

file_name = './output/sentiments.csv'
topic = "Trump"

consumer_key = settings['twitter_auth']['consumer_key']
consumer_secret = settings['twitter_auth']['consumer_secret']

access_token = settings['twitter_auth']['access_token']
access_token_secret = settings['twitter_auth']['access_token_secret']

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search(topic)

with open(file_name, mode='w', encoding='utf-16', newline="\n") as f:
    writer = csv.DictWriter(f, fieldnames=['Sentiment', 'Tweet text'])
    writer.writeheader()

    for tweet in public_tweets:
        analysis = TextBlob(tweet.text)
        label = "Positive" if analysis.sentiment.polarity >= 0 else "Negative"
        writer.writerow({'Sentiment': label, 'Tweet text': tweet.text})
