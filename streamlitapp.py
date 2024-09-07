import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
from bs4 import BeautifulSoup
import requests
from wordcloud import WordCloud
import nltk
from nltk import FreqDist
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import load_model

PAGE_HOME = "Home"
PAGE_PAGE1 = "Page 1"
PAGE_PAGE2 = "Page 2"
PAGE_PAGE3 = "Page 3"
# Define the label mapping for emotions
label_mapping = {0: 'anger', 1: 'dread', 2: 'joy', 3: 'sadness', 4: 'neutral', 5: 'surprise', 6: 'shame', 7: 'disgust'}
# Create a sidebar with navigation links
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", [PAGE_HOME,PAGE_PAGE2, 'UPLOAD_YOUR_CSV'],key="page_selector")

def home_page():
    st.title("Text Animation Example")

# Define a CSS class to apply the text color change effect
    st.markdown("""
        <style>
            .color-change {
                transition: color 0.5s;
            }
            .color-change:hover {
                color: red;
            }
        </style>
    """, unsafe_allow_html=True)

    # Apply the CSS class to a text element
    st.markdown('<p class="color-change">Hover over me to change color</p>', unsafe_allow_html=True)
    st.title("This is Sentix")
    st.write("Your go to sentiment analysis tool!")

    st.subheader("Analyze sentiment using link of different social media posts")
    page=st.sidebar.selectbox("Choose a social media platform", ["Instagram","Linkedin","Youtube"],key="platform_selector")
    def page4():
        plt.title("instagram")
    # Load your LSTM model, tokenizer, and other necessary data
        with open('./models/model.pkl', 'rb') as tokenizer_file:
            loaded_model = pickle.load(tokenizer_file)
        model = load_model('./models/emix.model.h5')
        with open('./models/tokenizer_sih.pkl', 'rb') as tokenizer_file:
            tokenizer = pickle.load(tokenizer_file)

        # Define custom functions for text cleaning and sentiment categorization
        def clean_text(tweet):
            # Add your custom text cleaning logic here
            tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet)
            tweet = re.sub(r"@\w+|#\w+", "", tweet)
            tweet = re.sub(r"[^a-zA-Z\s]", "", tweet)
            tweet = tweet.lower()
            tweet = re.sub(r"\s+", " ", tweet).strip()
            return tweet
        def preprocess_data(data):
            vocab_size = 5000  # Should match the vocab size used during model training
            onehot_repr = [one_hot(words, vocab_size) for words in [data]]
            sent_length = 50  # Should match the sequence length used during model training
            docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)
            return docs

        def categorize_sentiment(predictions):
            thresholds = [(0.97, "Highly Positive"), (0.8, "Positive"), (0.3, "Neutral"), (0.2, "Negative")]
            for threshold, label in thresholds:
                if predictions >= threshold:
                    return label
            return "Very Negative"

        
        # Streamlit UI
        

        # Text input box for user to enter a link
        link_input = st.text_input("Enter a link of instagram to analyze sentiment:")

        # Button to trigger scraping and sentiment analysis
        if st.button("Analyze Sentiment"):
            if link_input:
                with st.spinner("Predicting sentiment..."):
                    try:
                        # Scrape data from the provided link
                        response = requests.get(link_input)
                        soup = BeautifulSoup(response.content, "html.parser")
                        text_data = soup.get_text()
                        preprocessed_text = preprocess_data(text_data)
                        # Clean the scraped text
                        cleaned_text = clean_text(text_data)

                        # Tokenize and pad the text data
                        text_sequences = tokenizer.texts_to_sequences([cleaned_text])
                        text_sequences = pad_sequences(text_sequences, maxlen=100)

                        # Make predictions using the loaded model
                        predictions = loaded_model.predict(text_sequences)
                        e_prediction = model.predict(preprocessed_text)
                        print(e_prediction)
                        predicted_label = label_mapping[np.argmax(e_prediction)]
                        print(predicted_label)
                        # Categorize sentiment
                        sentiment = categorize_sentiment(predictions[0][0])
                        

                        # Display sentiment analysis results
                        st.subheader("Sentiment Analysis Results:")
                        #st.markdown(f"**Scraped Text:**\n{cleaned_text}")
                        st.markdown(f"**Sentiment:** {sentiment}")
                        st.markdown(f"**Emotion:** {predicted_label}")
                        print("Sentiment:", sentiment)
                        emoji_size=400
                        emoji_positive=(
                            f'<div style="display: flex; justify-content: center; align-items: center; height: 80vh;">'
                            f'<span style="font-size: {emoji_size}px;">😀</span>'
                            f'</div>'
                        )
                        emoji_negative=(
                            f'<div style="display: flex; justify-content: center; align-items: center; height: 80vh;">'
                            f'<span style="font-size: {emoji_size}px;">😞</span>'
                            f'</div>'
                        )
                        emoji_neutral=(
                            f'<div style="display: flex; justify-content: center; align-items: center; height: 80vh;">'
                            f'<span style="font-size: {emoji_size}px;">😐</span>'
                            f'</div>'
                        )
                        if(sentiment=='Positive'):
                            st.markdown(emoji_positive, unsafe_allow_html=True)
                        if(sentiment=='Negative'):
                            st.markdown(emoji_negative, unsafe_allow_html=True)
                        if(sentiment=='Neutral'):
                            st.markdown(emoji_neutral, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        print("Error:", str(e))
            else:
                st.warning("Please enter a link for scraping and sentiment analysis.")
    def page5():
        plt.title("Linkedin")
        loaded_model = load_model("./models/sentix_model.h5")
        model = load_model('./models/emix.model.h5')
        with open('./models/tokenizer_sih.pkl', 'rb') as tokenizer_file:
            tokenizer = pickle.load(tokenizer_file)

        # Define custom functions for text cleaning and sentiment categorization
        def clean_text(tweet):
            # Add your custom text cleaning logic here
            tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet)
            tweet = re.sub(r"@\w+|#\w+", "", tweet)
            tweet = re.sub(r"[^a-zA-Z\s]", "", tweet)
            tweet = tweet.lower()
            tweet = re.sub(r"\s+", " ", tweet).strip()
            return tweet
        def preprocess_data(data):
            vocab_size = 5000  # Should match the vocab size used during model training
            onehot_repr = [one_hot(words, vocab_size) for words in [data]]
            sent_length = 50  # Should match the sequence length used during model training
            docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)
            return docs

        def categorize_sentiment(predictions):
            thresholds = [(0.97, "Highly Positive"), (0.8, "Positive"), (0.3, "Neutral"), (0.2, "Negative")]
            for threshold, label in thresholds:
                if predictions >= threshold:
                    return label
            return "Very Negative"

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

        # Streamlit UI
        

        # Text input box for user to enter a link
        link_input = st.text_input("Enter a link of Linkedin to analyze sentiment:")

        # Button to trigger scraping and sentiment analysis
        if st.button("Analyze Sentiment"):
            if link_input:
                with st.spinner("Predicting sentiment..."):
                    try:
                        # Scrape data from the provided link
                        response = requests.get(link_input)
                        soup = BeautifulSoup(response.content, "html.parser")
                        text_data = soup.get_text()
                        start_index = text_data.find("Report this post")
                        end_index = text_data.find("Like")
                        substring = text_data[start_index+51:end_index]
                        cleaned_text = re.sub(r'\s+', ' ', substring)
                        preprocessed_text = preprocess_data(cleaned_text)
                        #preprocessed_text = preprocess_data(text_data)
                        # Clean the scraped text
                        #cleaned_text = clean_text(text_data)

                        # Tokenize and pad the text data
                        text_sequences = tokenizer.texts_to_sequences([cleaned_text])
                        text_sequences = pad_sequences(text_sequences, maxlen=100)

                        # Make predictions using the loaded model
                        predictions = loaded_model.predict(text_sequences)
                        e_prediction = model.predict(preprocessed_text)
                        print(e_prediction)
                        predicted_label = label_mapping[np.argmax(e_prediction)]
                        print(predicted_label)
                        # Categorize sentiment
                        sentiment = categorize_sentiment(predictions[0][0])
                        

                        # Display sentiment analysis results
                        st.subheader("Sentiment Analysis Results:")
                        #st.markdown(f"**Scraped Text:**\n{cleaned_text}")
                        st.markdown(f"**Sentiment:** {sentiment}")
                        st.markdown(f"**Emotion:** {predicted_label}")
                        
                        print("Sentiment:", sentiment)
                        emoji_size=400
                        emoji_positive=(
                            f'<div style="display: flex; justify-content: center; align-items: center; height: 80vh;">'
                            f'<span style="font-size: {emoji_size}px;">😀</span>'
                            f'</div>'
                        )
                        emoji_negative=(
                            f'<div style="display: flex; justify-content: center; align-items: center; height: 80vh;">'
                            f'<span style="font-size: {emoji_size}px;">😞</span>'
                            f'</div>'
                        )
                        emoji_neutral=(
                            f'<div style="display: flex; justify-content: center; align-items: center; height: 80vh;">'
                            f'<span style="font-size: {emoji_size}px;">😐</span>'
                            f'</div>'
                        )
                        if(sentiment=='Positive'):
                            st.markdown(emoji_positive, unsafe_allow_html=True)
                        if(sentiment=='Negative'):
                            st.markdown(emoji_negative, unsafe_allow_html=True)
                        if(sentiment=='Neutral'):
                            st.markdown(emoji_neutral, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        print("Error:", str(e))
            else:
                st.warning("Please enter a link for scraping and sentiment analysis.")
        
    def page6():
        plt.title("Youtube")
        loaded_model = load_model("./models/sentix_model.h5")
        model = load_model('./models/emix.model.h5')
        with open('./models/tokenizer_sih.pkl', 'rb') as tokenizer_file:
            tokenizer = pickle.load(tokenizer_file)

        # Define custom functions for text cleaning and sentiment categorization
        def clean_text(tweet):
            # Add your custom text cleaning logic here
            tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet)
            tweet = re.sub(r"@\w+|#\w+", "", tweet)
            tweet = re.sub(r"[^a-zA-Z\s]", "", tweet)
            tweet = tweet.lower()
            tweet = re.sub(r"\s+", " ", tweet).strip()
            return tweet
        def preprocess_data(data):
            vocab_size = 5000  # Should match the vocab size used during model training
            onehot_repr = [one_hot(words, vocab_size) for words in [data]]
            sent_length = 50  # Should match the sequence length used during model training
            docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)
            return docs

        def categorize_sentiment(predictions):
            thresholds = [(0.97, "Highly Positive"), (0.8, "Positive"), (0.3, "Neutral"), (0.2, "Negative")]
            for threshold, label in thresholds:
                if predictions >= threshold:
                    return label
            return "Very Negative"

       
        link_input = st.text_input("Enter a link of youtube to analyze sentiment:")

        # Button to trigger scraping and sentiment analysis
        if st.button("Analyze Sentiment"):
            if link_input:
                with st.spinner("Predicting sentiment..."):
                    try:
                        # Scrape data from the provided link
                        response = requests.get(link_input)
                        soup = BeautifulSoup(response.content, "html.parser")
                        description_element = soup.find('meta', {'name': 'description'})
                        if description_element:
                          video_description = description_element['content']

                        preprocessed_text = preprocess_data(video_description)
                        cleaned_text=clean_text(video_description)
                        text_sequences = tokenizer.texts_to_sequences([cleaned_text])
                        text_sequences = pad_sequences(text_sequences, maxlen=100)

                        # Make predictions using the loaded model
                        predictions = loaded_model.predict(text_sequences)
                        e_prediction = model.predict(preprocessed_text)
                        print(e_prediction)
                        predicted_label = label_mapping[np.argmax(e_prediction)]
                        print(predicted_label)
                        # Categorize sentiment
                        sentiment = categorize_sentiment(predictions[0][0])
                        

                        # Display sentiment analysis results
                        st.subheader("Sentiment Analysis Results:")
                        #st.markdown(f"**Scraped Text:**\n{cleaned_text}")
                        st.markdown(f"**Sentiment:** {sentiment}")
                        st.markdown(f"**Emotion:** {predicted_label}")
                        print("Sentiment:", sentiment)
                        emoji_size=400
                        emoji_positive=(
                            f'<div style="display: flex; justify-content: center; align-items: center; height: 80vh;">'
                            f'<span style="font-size: {emoji_size}px;">😀</span>'
                            f'</div>'
                        )
                        emoji_negative=(
                            f'<div style="display: flex; justify-content: center; align-items: center; height: 80vh;">'
                            f'<span style="font-size: {emoji_size}px;">😞</span>'
                            f'</div>'
                        )
                        emoji_neutral=(
                            f'<div style="display: flex; justify-content: center; align-items: center; height: 80vh;">'
                            f'<span style="font-size: {emoji_size}px;">😐</span>'
                            f'</div>'
                        )
                        if(sentiment=='Positive'):
                            st.markdown(emoji_positive, unsafe_allow_html=True)
                        if(sentiment=='Negative'):
                            st.markdown(emoji_negative, unsafe_allow_html=True)
                        if(sentiment=='Neutral'):
                            st.markdown(emoji_neutral, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        print("Error:", str(e))
            else:
                st.warning("Please enter a link for scraping and sentiment analysis.")
        

       
        
    if page == "Instagram":
        page4()
    elif page == "Linkedin":
        page5()
    elif page == "Youtube":
        page6()



def page2():
    tab1, tab2 = st.tabs(["Dataset", "Model"])
    tab1.subheader("Characteristics of Dataset")
    tab3,tab4,tab7=tab1.tabs(["Labels","Varied sizes","word cloud"])
    tab3.subheader("distribution of sentiments")
    tab3.image("./image/bar.png")
    tab3.image("./image/pie.png")
    tab4.subheader("Distribution showing the different lengths of tweets")
    tab4.image("./image/length.png")
    tab2.subheader("Model working")
    tab7.subheader("positive word cloud")
    tab7.image("./image/positive.png")
    tab7.subheader("negative word cloud")
    tab7.image("./image/negative.png")
    tab5,tab6=tab2.tabs(["Confusion Matrix","plots"])

def page3():
    st.title("Load a CSV file to predict output")
    with st.spinner("Loading..."):
        with open('./models/senti.pkl', 'rb') as model_file:
                loaded_model = pickle.load(model_file)
        #loaded_model = load_model("sentix_model.h5")
        # Load the tokenizer used during model training
        with open('./models/tokenizer_sih.pkl', 'rb') as tokenizer_file:
            tokenizer = pickle.load(tokenizer_file)

        def clean_text(text):
            cleaned_text = text.lower()
            cleaned_text = cleaned_text.replace('!', ' ')
            return cleaned_text

        st.sidebar.title("Settings")
        uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.write(df.head())
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
            
            # Class Distribution in 'label' column with 'viridis' color palette
            plt.title('Sentiment Distribution as BAR chart')
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.countplot(data=df, x='Sentiment', palette='viridis')
            plt.title('Sentiment Distribution in Training Dataset')
            plt.xlabel('Sentiment Labels')
            plt.ylabel('Count')

            # Display the plot using st.pyplot()
            st.pyplot(fig)
        # Create an interactive Altair chart
            
            plt.title('Sentiment Distribution as PIE chart')
            # Create a pie chart
            fig=plt.figure(figsize=(8, 6))
            sentiment_count = df["Sentiment"].value_counts()
            plt.pie(sentiment_count, labels=sentiment_count.index, autopct='%1.1f%%', shadow=False, startangle=140)
            

            # Display the pie chart using st.pyplot()
            st.pyplot(fig)
            st.title("Word Cloud for Positive Sentiment Tweets")  
            # Filter positive tweets
            pos_tweets = df[df["Sentiment"] == "Positive"]
            pos_tweets=pos_tweets.astype(str)
            print(pos_tweets)
        
            # Join positive tweets into a single string
            txt = " ".join(tweet.lower() for tweet in pos_tweets["Tweet"])

            # Generate the word cloud
            wordcloud = generate_wordcloud(txt)

            # Display the word cloud using st.image()
            st.pyplot(wordcloud)

            # Optionally, you can add other elements or text to your Streamlit app
            
            
            # Filter positive tweets
            h_pos_tweets = df[df["Sentiment"] == "Highly Positive"]
        
            # Join positive tweets into a single string
            txt = " ".join(tweet.lower() for tweet in h_pos_tweets["Tweet"])

            # Generate the word cloud
            wordcloud = generate_wordcloud(txt)

            # Display the word cloud using st.image()
            st.title("Word Cloud for Highly Positive Sentiment Tweets") 
            st.pyplot(wordcloud)

            # Optionally, you can add other elements or text to your Streamlit app
             

            # Filter positive tweets
            st.title("Word Cloud for Neutral Sentiment Tweets")
            neu_tweets = df[df["Sentiment"] == "Neutral"]
            neu_tweets = neu_tweets.astype(str)
            # Join positive tweets into a single string
            txt = " ".join(tweet.lower() for tweet in neu_tweets["Tweet"])

            # Generate the word cloud
            wordcloud = generate_wordcloud(txt)

            # Display the word cloud using st.image()
            st.pyplot(wordcloud)

            # Optionally, you can add other elements or text to your Streamlit app
            
            st.title("Word Cloud for Negative Sentiment Tweets")
            # Filter positive tweets
            neg_tweets = df[df["Sentiment"] == "Negative"]
            neg_tweets=neg_tweets.astype(str)
            # Join positive tweets into a single string
            txt = " ".join(tweet.lower() for tweet in neg_tweets["Tweet"])

            # Generate the word cloud
            wordcloud = generate_wordcloud(txt)

            # Display the word cloud using st.image()
            st.pyplot(wordcloud)

            # Optionally, you can add other elements or text to your Streamlit app
            
            st.title("Word Cloud for Very Negative Sentiment Tweets")
            # Filter positive tweets
            v_neg_tweets = df[df["Sentiment"] == "Very Negative"]
            v_neg_tweets=v_neg_tweets.astype(str)
            # Join positive tweets into a single string
            txt = " ".join(tweet.lower() for tweet in v_neg_tweets["Tweet"])

            # Generate the word cloud
            wordcloud = generate_wordcloud(txt)

            # Display the word cloud using st.image()
            st.pyplot(wordcloud)

            # Optionally, you can add other elements or text to your Streamlit app
            
            
            st.write("       ")
            st.write("       ")

            # Clean the text data using the custom cleaner
            text_data = text_data.apply(clean_text)
            wordcloud = generate_wordcloud(text_data)
            st.pyplot(wordcloud)

            st.title("Distribution of Tweet Length (Character Count)")
            df["Tweet"]=df["Tweet"].astype(str)
            df['tweet_length'] = df['Tweet'].apply(len)

            # Display a histogram plot of tweet lengths
            fig=plt.figure(figsize=(8, 6))
            sns.histplot(df['tweet_length'], bins=50)
            plt.title("Distribution of Text Length (Character Count)")
            plt.xlabel("Text Length")
            plt.ylabel("Count")

            # Display the plot using st.pyplot()
            st.pyplot(fig)


            # Collect all words from all tweets into a single list
            all_words = []
            for t in df['Tweet']:
                all_words.extend(t.split())


            # Calculate and display the number of unique words
            unique_word_count = len(set(all_words))
            st.write(f"Number of unique words: {unique_word_count}")

            # Frequency Distribution
            freq_dist = FreqDist(all_words)

            # Create a Streamlit app
            st.title('Top 50 Most Common Words')

            # Plot the top 50 most common words
            fig=plt.figure(figsize=(16, 5))
            plt.title('Top 50 most common words')
            plt.xticks(fontsize=15)

            # Display the frequency distribution plot using st.pyplot()
            freq_dist.plot(50, cumulative=False)
            st.pyplot(fig)


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

if __name__ == "__main__":
    if page == PAGE_HOME:
        home_page()
    #elif page == PAGE_PAGE1:
     #   page1()
    elif page == PAGE_PAGE2:
        page2()
    elif page == 'UPLOAD_YOUR_CSV':
        page3()

