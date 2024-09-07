
import pickle
import uvicorn
from pydantic import BaseModel

from tensorflow import keras
import numpy as np
from fastapi import FastAPI, HTTPException

# Import the necessary functions and classes from your model code
from tensorflow.keras.preprocessing.text import one_hot

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import requests
from bs4 import BeautifulSoup
class Item(BaseModel):
    link: str
# Load the tokenizer
with open("./models/tokenizer_sih.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# Load your LSTM model (assuming it's defined elsewhere)
lstm = load_model("./models/sentix_model.h5")
model = load_model('./models/emix.model.h5')  # Replace with the path to your saved model

# Define the label mapping for emotions
label_mapping = {0: 'anger', 1: 'fear', 2: 'joy', 3: 'sadness', 4: 'neutral', 5: 'surprise', 6: 'shame', 7: 'disgust'}

def clean_tweet(tweet):
    # Remove URLs
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet)
    
    # Remove mentions and hashtags
    tweet = re.sub(r"@\w+|#\w+", "", tweet)
    
    # Remove special characters and numbers
    tweet = re.sub(r"[^a-zA-Z\s]", "", tweet)
    
    # Convert to lowercase
    tweet = tweet.lower()
    
    # Remove extra whitespaces
    tweet = re.sub(r"\s+", " ", tweet).strip()
    
    return tweet

def scrape_text_from_link(link):
    try:
        response = requests.get(link)
        soup = BeautifulSoup(response.content, "html.parser")
        # Assuming the text is contained within <p> elements
        text = soup.get_text()
        text = clean_tweet(text)
        return text
    except Exception as e:
        return str(e)

def preprocess_data(data):
    vocab_size = 5000  # Should match the vocab size used during model training
    onehot_repr = [one_hot(words, vocab_size) for words in [data]]
    sent_length = 50  # Should match the sequence length used during model training
    docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)
    return docs
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}
@app.get("/predict")
def predict(link: str):
    scraped_text = scrape_text_from_link(link)
    print(scraped_text)
    if not scraped_text:
        return {"error": "Unable to scrape text from the provided link"}

    X = clean_tweet(scraped_text)
    preprocessed_text = preprocess_data(scraped_text)

        # Make a prediction using the loaded model
    e_prediction = model.predict(preprocessed_text)

        # Get the predicted emotion label
    predicted_label = label_mapping[np.argmax(e_prediction)]
    # Tokenize the text and convert to sequence of token IDs
    X_seq = tokenizer.texts_to_sequences([X])
    
    # Pad the sequence of token IDs
    X_seq = pad_sequences(X_seq, maxlen=100)  # Use X_seq instead of X
    
    # Assuming you have loaded the lstm model earlier
    custom_sentiment_prob = lstm.predict(X_seq)[0][0]

    # Define custom sentiment categories and thresholds
    thresholds = [(0.97, "Highly Positive"), (0.8, "Positive"), (0.3, "Neutral"), (0.2, "Negative")]

    # Initialize custom_sentiment to "Highly Negative" by default
    custom_sentiment = "Highly Negative"

    # Determine sentiment based on the probability and thresholds
    for threshold, sentiment_label in thresholds:
        if custom_sentiment_prob >= threshold:
            custom_sentiment = sentiment_label
            break  # Break the loop once a matching threshold is found

    # Print the sentiment and probability score
    print(f"Sentiment: {custom_sentiment}")
    print(custom_sentiment_prob)
    return {"text": scraped_text, "prediction": float(custom_sentiment_prob), "sentiment": custom_sentiment,"emotion": predicted_label}

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)