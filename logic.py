from googleapiclient.discovery import build
import pandas as pd
import nltk
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import re
import string
import warnings
from dotenv import load_dotenv

# Download NLTK resources, VADER lexicon for sentiment analysis, etc.
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nlp = spacy.load("en_core_web_sm")
nltk.download('vader_lexicon')

# Suppress all warnings
warnings.filterwarnings("ignore")

# Setting up YouTube API
load_dotenv()
import os
# Retrieve the API key from the environment variable
api_key = os.getenv('YOUTUBE_API_KEY')

youtube = build('youtube', 'v3', developerKey=api_key)

def retrieve_comments(video_id):
    comments = []
    nextPageToken = None
    while True:
        # Calling API to retrieve comment threads
        response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            pageToken=nextPageToken if nextPageToken else ''
        ).execute()

        # Extracting comment data
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'updatedAt': comment['updatedAt'],
                'likeCount': comment['likeCount'],
                'author': comment['authorDisplayName'],
                'text': comment['textDisplay'],
                'published_at': comment['publishedAt']
            })

        # Checking if there are more comments to fetch
        nextPageToken = response.get('nextPageToken')
        if not nextPageToken:
            break
    return comments


# Initializing the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()


def analyze_sentiment(comment):
    # Analyzing sentiment of the comment using VADER
    sentiment_scores = sid.polarity_scores(comment)

    # Classifying the sentiment as positive, negative, or neutral based on compound score
    if sentiment_scores['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


def plot_sentiment_distribution_pie(df):
    # Counting the occurrences of each sentiment category
    sentiment_counts = df['sentiment'].value_counts()

    # Calculating the percentage of each sentiment category
    total_comments = len(df)
    sentiment_percentages = (sentiment_counts / total_comments) * 100

    # Creating a data frame for plotting
    plot_data = pd.DataFrame({'Sentiment': sentiment_percentages.index,
                              'Percentage': sentiment_percentages.values,
                              'Count': sentiment_counts.values})
    
    return plot_data


def remove_html_tags(text):
    # Removeing HTML tags using beautiful soup
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text(separator=" ")
    return cleaned_text


def replace_contractions(text):
    return text.replace("nt", " not")

def preprocess_text(text):
    # Removing emojis
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # Converting text to lowercase
    text = text.lower()

    # Removing HTML links
    text = re.sub(r'http\S+', '', text)

    # Removing multiple consecutive spaces
    text = re.sub(r'\s+', ' ', text)

    # Tokenizing the text
    tokens = word_tokenize(text)

    # Removing punctuation
    table = str.maketrans('', '', string.punctuation)
    tokens = [token.translate(table) for token in tokens]

    # Replace contraction
    tokens = [replace_contractions(text) for text in tokens]

    # Removing stopwords and short words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]

    # Performing NER
    doc = nlp(' '.join(tokens))
    tokens_to_keep = []
    for token in doc:
        # Keeping named entities and tokens longer than 1 character
        if token.ent_type_ != '' or len(token) > 1:
            tokens_to_keep.append(token.text)

    processed_text = ' '.join(tokens_to_keep)

    return processed_text