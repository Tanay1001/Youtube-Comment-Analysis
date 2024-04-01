import streamlit as st
import logic
import pandas as pd
import nltk
import spacy
import plotly.express as px
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
import emoji
from wordcloud import WordCloud
import calendar
from textblob import TextBlob
from langdetect import detect
import langcodes
import warnings

# Download NLTK resources, VADER lexicon for sentiment analysis, etc.
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nlp = spacy.load("en_core_web_sm")
nltk.download('vader_lexicon')

# Suppress all warnings
warnings.filterwarnings("ignore")

# Title and simple text
st.title('YOUTUBE COMMENT ANALYSIS')
st.write('-By Tanay')


# Eg. Sample Video ID: 9G69n11o3z8


# Define a function for plotting top words for each topic
def plot_top_words(model, feature_names, n_top_words):
    n_topics = model.n_components
    n_top_words = min(n_top_words, len(feature_names))

    # Create a new figure with subplots arranged in a single row
    fig, axes = plt.subplots(1, n_topics, figsize=(16, 6))

    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx + 1}", fontdict={'fontsize': 20})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=14)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)

    # Adjust layout
    fig.subplots_adjust(wspace=0.4)

    # Return the figure object
    return fig

def on_change_callback(video_id):
    comments = logic.retrieve_comments(video_id)
    df = pd.DataFrame(comments)

    st.write("Analysis Report Generated:")

    st.title("Raw Comments Data")
    st.dataframe(df)

    
    
    # Total comments
    total_comments = len(df)

    # Total words
    total_unique_words = len(set(word for comment in df['text'] for word in comment.split()))

    # Regular expression pattern to match URLs
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Extracting all URLs from the comments into a list
    all_links = [re.findall(url_pattern, comment) for comment in df['text']]
    all_links_flat = [link for sublist in all_links for link in sublist]

    # Total number of links shared
    total_links_shared = len(all_links_flat)
    

    # Tokenize each comment into words and calculate the number of words
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    average_word_count = round(df['word_count'].mean(),2)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.header("Total Comments")
        st.title(total_comments)
    with col2:
        st.header("Total Unique Words")
        st.title(total_unique_words)
    with col3:
        st.header("Total Links Shared")
        st.title(total_links_shared)
    with col4:
        st.header("Average Comment Length")
        st.title(average_word_count)
    
    
    # Performing sentiment analysis on each comment
    df['sentiment'] = df['text'].apply(logic.analyze_sentiment)

    plot_data = logic.plot_sentiment_distribution_pie(df)

    # Plotting a pie chart
    fig1 = px.pie(plot_data, names='Sentiment', values='Percentage',
                 hover_data={'Count': True},
                 labels={'Sentiment': 'Sentiment', 'Percentage': 'Percentage (%)', 'Count': 'Count'})
    
    st.title("Sentiment Distribution")
    st.plotly_chart(fig1)

    # Removing HTML tags and links from the text
    df['cleaned_text'] = df['text'].apply(logic.remove_html_tags)

    # Preprocessing the cleaned text data
    df['processed_text'] = df['cleaned_text'].apply(logic.preprocess_text)
        
    
    # Converting the preprocessed comments into a document-term matrix
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(df['processed_text'])

    # Applying LDA
    lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
    lda_model.fit(dtm)

    feature_names = vectorizer.get_feature_names_out()

    fig_axes_list = plot_top_words(lda_model, feature_names, n_top_words=10)

    # Streamlit app
    st.title("Top Words for Each Topic")


    if 'fig_axes_list' in locals():
        st.pyplot(fig_axes_list)

    # Extracting all emojis into a list
    emojis = []
    for message in df['text']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])
    
    # Counting the frequency of each emoji
    emoji_freq = Counter(emojis)

    # Top 10 emojis with their frequencies
    top_10_emojis = emoji_freq.most_common(10)
    top_10_df = pd.DataFrame(top_10_emojis, columns=['Emoji', 'Frequency'])
    st.title("Top 10  Emojis")
    st.dataframe(top_10_df)


    # Extracting year, month, day, and time components
    df['updatedAt'] = pd.to_datetime(df['updatedAt'])
    df['year'] = df['updatedAt'].dt.year
    df['month'] = df['updatedAt'].dt.month
    df['day'] = df['updatedAt'].dt.day
    df['time'] = df['updatedAt'].dt.strftime('%H:%M:%S')


    # Generating the word cloud
    st.title("Word Cloud")
    text = ' '.join(df['processed_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Displaying the figure with the wordcloud
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(fig)



    # Top 20 words with frequencies
    st.title("Top 20 words with frequencies")
    words = []
    for message in df['processed_text']:
        for word in message.lower().split():
            words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20), columns=['Word', 'Frequency'])
    st.dataframe(most_common_df)


    # Couting total number of comments by month
    timeline = df.groupby(['year', 'month']).count()['text'].reset_index()

    # Converting year column to string data type
    timeline['year'] = timeline['year'].astype(str)

    # Converting month numbers to month names
    timeline['month'] = timeline['month'].apply(lambda x: calendar.month_name[x])

    # Concatenating month and year columns
    timeline['time'] = timeline['month'] + "-" + timeline['year'].astype(str)


    # Plotting monthly frequencies
    fig, ax = plt.subplots()
    st.title("Monthly Comments")
    ax.plot(timeline['time'], timeline['text'], color='green')
    ax.set_xticklabels(ax.get_xticks(), rotation=90, ha='right')
    st.pyplot(fig)



    # Plotting daily frequencies
    fig, ax = plt.subplots()
    df['only_date'] = df['updatedAt'].dt.date
    daily_timeline = df.groupby('only_date').count()['text'].reset_index()
    ax.plot(daily_timeline['only_date'], daily_timeline['text'], color='black')
    plt.xticks(rotation='vertical')
    st.title("Daily Comments")
    st.pyplot(fig)



    # Calculate sentiment scores for each text using text blob
    df['sentiment_score'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Grouping by month and calculating total comments and average sentiment score
    monthly_stats = df.groupby(['year', 'month']).agg({'text': 'count', 'sentiment_score': 'mean'}).reset_index()
    monthly_stats.rename(columns={'text': 'total_comments'}, inplace=True)

    fig, ax1 = plt.subplots()

    # Plotting total monthly comments
    ax1.bar(monthly_stats.index, monthly_stats['total_comments'], color='skyblue')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Total Comments', color='skyblue')

    # X-axis for sentiment score plot
    ax2 = ax1.twinx()
    ax2.plot(monthly_stats.index, monthly_stats['sentiment_score'], color='red', marker='o')
    ax2.set_ylabel('Average Sentiment Score', color='red')
    ax1.set_xticks(monthly_stats.index)
    ax1.set_xticklabels(monthly_stats.apply(lambda x: f"{calendar.month_name[int(x['month'])]}-{int(x['year'])}", axis=1), rotation=90, ha='right')
    plt.tight_layout()
    st.title("Total Monthly Comments and Average Sentiment Score")
    st.pyplot(fig)



    def detect_language(text):
        try:
            lang = detect(text)
            return lang
        except:
            return None

    df['language'] = df['text'].apply(detect_language)
    language_counts = df['language'].value_counts()



    # Top 10 languages
    top_10_languages = language_counts.head(10)

    language_data = []

    # Iterate over top 10 languages and fetch language name
    for language_code, count in top_10_languages.items():
        try:
            language_name = langcodes.Language(language_code).language_name('en')
        except ValueError:
            language_name = f"Unknown Language ({language_code})"
        language_data.append({'Language': language_name, 'Count': count})

    # Convert list of dictionaries to DataFrame
    language_df = pd.DataFrame(language_data)



    # Streamlit display
    st.write('Top 10 Language Distribution:')
    st.table(language_df)



    # Calculating and plotting average likes per comment for negative, positive, and neutral comments
    negative_likes = df[df['sentiment'] == 'Negative']['likeCount'].sum()
    positive_likes = df[df['sentiment'] == 'Positive']['likeCount'].sum()
    neutral_likes = df[df['sentiment'] == 'Neutral']['likeCount'].sum()

    average_negative_likes = negative_likes / len(df[df['sentiment'] == 'Negative'])
    average_positive_likes = positive_likes / len(df[df['sentiment'] == 'Positive'])
    average_neutral_likes = neutral_likes / len(df[df['sentiment'] == 'Neutral'])

    sentiments = ['Negative', 'Positive', 'Neutral']
    total_likes = [negative_likes, positive_likes, neutral_likes]
    average_likes = [average_negative_likes, average_positive_likes, average_neutral_likes]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Total Likes
    ax1.bar(sentiments, total_likes, color=['red', 'green', 'blue'])
    ax1.set_title('Total Likes on Comments by Sentiment')
    ax1.set_ylabel('Total Likes')

    # Average Likes
    ax2.bar(sentiments, average_likes, color=['red', 'green', 'blue'])
    ax2.set_title('Average Likes per Comment by Sentiment')
    ax2.set_ylabel('Average Likes')

    plt.tight_layout()

    # Show plot with Streamlit
    st.pyplot(fig)



video_id = st.text_input("Enter Youtube Video ID:")
button = st.button("Generate Report")

if button and len(str(video_id)) > 1:
    on_change_callback(video_id)
elif button and len(str(video_id)) <= 1:
    st.write("Error: Please Enter Video ID")