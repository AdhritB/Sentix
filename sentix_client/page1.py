import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from bs4 import BeautifulSoup
import requests
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot


with open('../models/tokenizer_emo.pkl', 'rb') as tokenizer_file:
    tokenizer_emo = pickle.load(tokenizer_file)
with open('../models/tokenizer_sih.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)
with open('../models/model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


# Define custom functions for text cleaning and sentiment categorization
def clean_text(tweet):
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet)
    tweet = re.sub(r"@\w+|#\w+", "", tweet)
    tweet = re.sub(r"[^a-zA-Z\s]", "", tweet)
    tweet = tweet.lower()
    tweet = re.sub(r"\s+", " ", tweet).strip()
    return tweet

def preprocess_data(data):
    vocab_size = 5000  # Should match the vocab size used during model training
    onehot_repr = [one_hot(words, vocab_size) for words in [data]]
    sent_length = 100  # Should match the sequence length used during model training
    docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)
    return docs

def categorize_sentiment(predictions):
    thresholds = [(0.97, "Highly Positive"), (0.8, "Positive"), (0.3, "Neutral"), (0.2, "Negative")]
    for threshold, label in thresholds:
        if predictions >= threshold:
            return label
    return "Very Negative"

# Function to perform emotion analysis (you can use a more advanced model here)
def predict_emotion(text):
    # Load the trained model
    with open('../models/emo_model.pkl', 'rb') as model_file:
            emo_model = pickle.load(model_file)  # Replace with the path to your trained model

    # Text normalization
    normalized_text = text

    # Tokenize and pad the input text
    tokenizer = Tokenizer(oov_token='UNK')
    tokenizer.fit_on_texts([normalized_text])
    sequences = tokenizer.texts_to_sequences([normalized_text])
    padded_sequences = pad_sequences(sequences, maxlen=229, truncating='pre')

    # Make the prediction
    predicted_probs = emo_model.predict(padded_sequences)
    predicted_class = np.argmax(predicted_probs, axis=-1)

    # Map the predicted class back to the original emotion label (use the LabelEncoder)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(['anger 😡','fear 😨','joy 🙂','love 🥰','sadness 😓','surprise 😙 '])  # Replace with your class labels
    predicted_emotion = label_encoder.inverse_transform(predicted_class)

    return predicted_emotion[0], predicted_probs[0]

# Streamlit app
def app():
    
    st.title("Social Media Sentiment Analysis")

    # Create tabs for different social media platforms
    selected_tab = st.selectbox("Select a Social Media Platform", ["YouTube", "Instagram", "LinkedIn"])

    # Input box for the user to enter a link
    link = st.text_input(f"Enter a {selected_tab} Link:")

    if st.button("Analyze Sentiment"):
        if link:
            with st.spinner("Fetching and analyzing data..."):
                try:
                    # Scrape data from the provided link
                    response = requests.get(link)
                    soup = BeautifulSoup(response.content, "html.parser")
                    if selected_tab == "YouTube":
                        # Extract text from YouTube video description
                        description_element = soup.find('meta', {'name': 'description'})
                        if description_element:
                            video_description = description_element['content']
                            cleaned_text = clean_text(video_description)
                        # Perform sentiment analysis
                            text_sequences = tokenizer.texts_to_sequences([cleaned_text])
                            text_sequences = pad_sequences(text_sequences, maxlen=100)
                            predictions = loaded_model.predict(text_sequences)
                            sentiment = categorize_sentiment(predictions[0][0])
                            print(sentiment)
                            # Perform emotion analysis
                            # emotion = perform_emotion_analysis(cleaned_text)

                            st.subheader("Sentiment:")
                            st.write(sentiment)
                            st.subheader("Polarity:")
                            st.write(predictions[0][0])
                            if sentiment =='Neutral' and predicted_emotion =='sadness':
                                predicted_emotion = 'ambiguous 🤓 '
                            elif sentiment == 'Positive' and predicted_emotion =='sadness':
                                predicted_emotion = 'satisfaction 🤓 '
                            st.subheader("Emotion:")
                            st.write(predicted_emotion)
                            st.subheader("Polarity:")
                            label_class= np.array(['anger 😡','fear 😨','joy 🙂','love 🥰','sadness 😓','surprise 😙 '])
                            data = pd.DataFrame({'Emotion Label': label_class,'Predicted Probs': predicted_probs})
                            st.table(data)
                    elif selected_tab == "Instagram":
                        # Extract text from Instagram post (you may need to adapt this for Instagram)
                        text_data = soup.get_text()
                        cleaned_text = clean_text(text_data)
                        print(cleaned_text)
                        # Perform sentiment analysis
                        text_sequences = tokenizer.texts_to_sequences([cleaned_text])
                        text_sequences = pad_sequences(text_sequences, maxlen=100)
                        predictions = loaded_model.predict(text_sequences)
                        sentiment = categorize_sentiment(predictions)
                        predicted_emotion, predicted_probs = predict_emotion(cleaned_text)

                        st.subheader("Sentiment:")
                        st.write(sentiment)
                        st.subheader("Polarity:")
                        st.write(predictions[0][0])
                        if sentiment =='Neutral' and predicted_emotion =='sadness':
                            predicted_emotion = 'ambiguous 🤓 '
                        elif sentiment == 'Positive' and predicted_emotion =='sadness':
                            predicted_emotion = 'satisfaction 🤓 '
                        st.subheader("Emotion:")
                        st.write(predicted_emotion)
                        st.subheader("Polarity:")
                        label_class= np.array(['anger 😡','fear 😨','joy 🙂','love 🥰','sadness 😓','surprise 😙 '])
                        data = pd.DataFrame({'Emotion Label': label_class,'Predicted Probs': predicted_probs})
                        st.table(data)
                    elif selected_tab == "LinkedIn":
                        # Extract text from LinkedIn post (you may need to adapt this for LinkedIn)
                        text_data = soup.get_text()
                        start_index = text_data.find("Report this post")
                        end_index = text_data.find("Like")
                        substring = text_data[start_index + 51:end_index]

                        cleaned_text = re.sub(r'\s+', ' ', substring)
                        # Perform sentiment analysis
                        text_sequences = tokenizer.texts_to_sequences([cleaned_text])
                        text_sequences = pad_sequences(text_sequences, maxlen=100)
                        predictions = loaded_model.predict(text_sequences)
                        sentiment = categorize_sentiment(predictions[0][0])
                        
                        predicted_emotion, predicted_probs = predict_emotion(cleaned_text)

                            # Display results
                        st.subheader("Sentiment:")
                        st.write(sentiment)
                        st.subheader("Polarity:")
                        st.write(predictions[0][0])
                        if sentiment =='Neutral' and predicted_emotion =='sadness':
                            predicted_emotion = 'ambiguous 🤓 '
                        elif sentiment == 'Positive' and predicted_emotion =='sadness':
                            predicted_emotion = 'satisfaction 🤓 '
                        st.subheader("Emotion:")
                        st.write(predicted_emotion)
                        st.subheader("Polarity:")
                        label_class= np.array(['anger 😡','fear 😨','joy 🙂','love 🥰','sadness 😓','surprise 😙 '])
                        data = pd.DataFrame({'Emotion Label': label_class,'Predicted Probs': predicted_probs})
                        st.table(data)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                        
                    
        else:
            st.warning("Please enter a link for scraping and sentiment analysis.")

if __name__ == "__main__":
    app()
