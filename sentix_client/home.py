# home.py
import streamlit as st
import time
# Adjust the sleep duration to control the typing speed


def app():
    st.title("Welcome to SENTIX")
    st.subheader('Your Sentiment analysing tool')
    st.write('Sentix is a social media sentiment analyzer tool to monitor social media presence. Our tool utilizes LSTM models, a powerful deep learning technique, to accurately analyze and classify sentiment in social media posts and comments.\n\n ')
    bottom_right_content = "Developed by Team Blinders"
    bottom_right_css = """
    <style>
    .bottom-right {
        position: absolute;
        bottom: 10px;
        right: 10px;
    }
    </style>
    """

    # Inject the custom CSS into the Streamlit app
    st.markdown(bottom_right_css, unsafe_allow_html=True)
    st.write(f'<div class="bottom-right">{bottom_right_content}</div>', unsafe_allow_html=True)