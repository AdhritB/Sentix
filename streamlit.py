# import streamlit as st
# import requests
# from wordcloud import WordCloud  # Import WordCloud here if it's not already imported
# import matplotlib.pyplot as plt
# import pandas as pd
# # Streamlit UI
# st.title("Sentiment Analysis with FastAPI and Keras")

# # Text input box for user input
# user_input = st.text_area("Enter text:", "")

# # Button to trigger the prediction
# if st.button("Predict Sentiment"):
#     if user_input:
#         # Show a spinner while waiting for the response
#         with st.spinner("Predicting sentiment..."):
#             # Define the FastAPI endpoint URL
#             endpoint_url = "http://localhost:8000/predict"  # Update with your FastAPI server's URL

#             # Send a GET request to the FastAPI endpoint with the user's input
#             response = requests.get(endpoint_url, params={"link": user_input})

#             if response.status_code == 200:
#                 result = response.json()

#                 # Display the sentiment prediction and probability with emojis
#                 st.subheader("Sentiment Prediction:")
#                 sentiment = result['sentiment']
#                 prediction = result['prediction']
#                 emotion = result['emotion']
#                 emoji = "😃" if sentiment == "positive" else "😞" if sentiment == "negative" else "😐"
#                 st.markdown(f"**Text:** {result['text']}")
#                 st.markdown(f"**Sentiment:** {sentiment} {emoji}")
#                 st.markdown(f"**Emotion:** {emotion} ")
#                 st.markdown(f"**Probability:** {prediction:.6f}")

#                 # Generate and display the Word Cloud
#                 st.subheader("Word Cloud of Tweet Text:")
#                 wordcloud = WordCloud(width=800, height=400, background_color='white').generate(result['text'])
#                 plt.figure(figsize=(11, 10))
#                 plt.imshow(wordcloud, interpolation='bilinear')
#                 plt.title("Word Cloud of Tweet Text")
#                 plt.axis('off')
#                 st.pyplot(plt)
#             else:
#                 st.error(f"Error: {response.status_code} - Unable to get sentiment prediction.")
#     else:
#         st.warning("Please enter some text for sentiment prediction.")


import streamlit as st

# Function to embed a social media link using an iframe
def embed_social_media_link(link):
    # Use HTML to create an iframe element with the provided link
    iframe_code = f'<iframe src="{link}" width="500" height="300" frameborder="0"></iframe>'
    st.write(iframe_code, unsafe_allow_html=True)

# Streamlit app
st.title("Embed Social Media Link")

# Example: Embed a Twitter tweet
twitter_tweet_link = 'https://www.linkedin.com/posts/aditya-datta-9152ba1a8_the-data-centric-approach-to-ai-techfastly-activity-6948139366393221121-bjmB?utm_source=share&utm_medium=member_android'
st.write("Embedding a Twitter tweet:")
embed_social_media_link(twitter_tweet_link)

# You can add more social media links as needed
