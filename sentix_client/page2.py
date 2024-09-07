# home.py
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np

import pickle

from wordcloud import WordCloud

from nltk import FreqDist
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import load_model

# Define a custom CSS style
custom_css = """
<style>
/* Improve sidebar styling */
.sidebar .sidebar-content {
    background-color: #333;
    padding: 20px;
    border-radius: 10px;
}

/* Style main content */
.main {
    padding: 20px;
    background-color: #f0f0f0;
    border-radius: 10px;
}

/* Style titles and headers */
h1 {
    color: #333;
    font-size: 32px;
}

h2 {
    color: #555;
    font-size: 24px;
}

/* Style buttons */
div[data-testid="stButton"] button {
    background-color: #0077b6;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    cursor: pointer;
}

div[data-testid="stButton"] button:hover {
    background-color: #005B91;
}

/* Style data tables */
.dataframe {
    border: 1px solid #ddd;
    border-collapse: collapse;
}

.dataframe th, .dataframe td {
    padding: 8px;
    border: 1px solid #ddd;
    text-align: left;
}

/* Style word cloud images */
img.wordcloud {
    width: 100%;
    height: auto;
}

</style>
"""
def generate_wordcloud(text_data):
        # Join the text data into a single string
        text = ' '.join(text_data)

        # Generate the WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        # Display the WordCloud using Matplotlib
        plt.figure(figsize=(11, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title("Word Cloud of Tweet Text")
        plt.axis('off')
        return plt
def app():
    st.title("Upload your CSV 📊📋📈")
    with st.spinner("Please wait while our model understands your data..."):
        with open('../models/senti.pkl', 'rb') as model_file:
                loaded_model = pickle.load(model_file)
            #loaded_model = load_model("sentix_model.h5")
        # Load the tokenizer used during model training
        with open('../models/tokenizer_sih.pkl', 'rb') as tokenizer_file:
            tokenizer = pickle.load(tokenizer_file)

        def clean_text(text):
            cleaned_text = text.lower()
            cleaned_text = cleaned_text.replace('!', ' ')
            return cleaned_text

        st.sidebar.title("Settings")
        uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.subheader("Data Preview")
            st.sidebar.write(df.head())
            df = df.rename(columns={df.columns[1]: "Tweet"})
            # Assuming your text data column is named 'Tweet'
            text_data = df['Tweet'].astype(str)

            # Tokenize and pad the text data
            text_sequences = tokenizer.texts_to_sequences(text_data)
            text_sequences = pad_sequences(text_sequences, maxlen=100)  # Assuming max_sequence_length is 100

                # Make predictions using the loaded model
            predictions = loaded_model.predict(text_sequences)

                # Add predictions to the DataFrame
            df['Predicted_Sentiment'] = predictions
            thresholds = [(0.97, "Highly Positive"), (0.8, "Positive"), (0.3, "Neutral"), (0.2, "Negative")]

            def categorize_sentiment(predictions):
                for threshold, label in thresholds:
                    if predictions >= threshold:
                        return label
                return "Very Negative"

            df['Sentiment'] = predictions
            df['Sentiment'] = df['Sentiment'].apply(categorize_sentiment)
            st.write("Predictions and Sentiments:")
            st.write(df[['Tweet', 'Sentiment','Predicted_Sentiment']])
                
            st.markdown('Sentiment Distribution as BAR chart')
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.countplot(data=df, x='Sentiment', palette='viridis')
            plt.title('Sentiment Distribution in Training Dataset')
            plt.xlabel('Sentiment Labels')
            plt.ylabel('Count')


            st.pyplot(fig)

                
            st.markdown('Sentiment Distribution as PIE chart')

            fig=plt.figure(figsize=(8, 6))
            sentiment_count = df["Sentiment"].value_counts()
            plt.pie(sentiment_count, labels=sentiment_count.index, autopct='%1.1f%%', shadow=False, startangle=140)
            st.pyplot(fig)
            st.markdown("Word Cloud for Positive Sentiment Posts")  

            pos_tweets = df[df["Sentiment"] == "Positive"]
            pos_tweets=pos_tweets.astype(str)
            print(pos_tweets)
            txt = " ".join(tweet.lower() for tweet in pos_tweets["Tweet"])
            wordcloud = generate_wordcloud(txt)


            st.pyplot(wordcloud)
            h_pos_tweets = df[df["Sentiment"] == "Highly Positive"]
            txt = " ".join(tweet.lower() for tweet in h_pos_tweets["Tweet"])
            wordcloud = generate_wordcloud(txt)
            st.markdown("Word Cloud for Highly Positive Sentiment Posts") 
            st.pyplot(wordcloud)

            st.markdown("Word Cloud for Neutral Sentiment Posts")
            neu_tweets = df[df["Sentiment"] == "Neutral"]
            neu_tweets = neu_tweets.astype(str)
                # Join positive tweets into a single string
            txt = " ".join(tweet.lower() for tweet in neu_tweets["Tweet"])

            wordcloud = generate_wordcloud(txt)
            st.pyplot(wordcloud)

        
                
            st.markdown("Word Cloud for Negative Sentiment Posts")
            neg_tweets = df[df["Sentiment"] == "Negative"]
            neg_tweets=neg_tweets.astype(str)
            txt = " ".join(tweet.lower() for tweet in neg_tweets["Tweet"])
            wordcloud = generate_wordcloud(txt)
            st.pyplot(wordcloud)

            st.markdown("Word Cloud for Very Negative Sentiment Posts")
                
            v_neg_tweets = df[df["Sentiment"] == "Very Negative"]
            v_neg_tweets=v_neg_tweets.astype(str)
            # Join positive tweets into a single string
            txt = " ".join(tweet.lower() for tweet in v_neg_tweets["Tweet"])

            wordcloud = generate_wordcloud(txt)

            st.pyplot(wordcloud)

            st.write("       ")
            st.write("       ")

            st.markdown("Word Cloud for Overall Data")
            text_data = text_data.apply(clean_text)
            wordcloud = generate_wordcloud(text_data)
            st.pyplot(wordcloud)

            st.markdown("Distribution of Tweet Length (Character Count)")
            df["Tweet"]=df["Tweet"].astype(str)
            df['tweet_length'] = df['Tweet'].apply(len)

            # Display a histogram plot of tweet lengths
            fig=plt.figure(figsize=(8, 6))
            sns.histplot(df['tweet_length'], bins=50)
            plt.title("Distribution of Text Length (Character Count)")
            plt.xlabel("Text Length")
            plt.ylabel("Count")
            st.pyplot(fig)


            # Collect all words from all tweets into a single list
            all_words = []
            for t in df['Tweet']:
                all_words.extend(t.split())


            # Calculate and display the number of unique words
            unique_word_count = len(set(all_words))
            st.markdown(f"Number of unique words: {unique_word_count}")

            freq_dist = FreqDist(all_words)

            st.markdown('Top 50 Most Common Words')

            # Plot the top 50 most common words
            fig=plt.figure(figsize=(20, 5))
            plt.title('Top 50 most common words')
            plt.xticks(fontsize=15)
            
            freq_dist.plot(50, cumulative=False)
            st.pyplot(fig)


    