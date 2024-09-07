# navigation.py
import streamlit as st
from home import app as home_app
from page1 import app as page1_app
from page2 import app as page2_app


def main():
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.selectbox("Go to", ["What's SENTIX", "Review Your Socials", "Upload Your Data"])


    if selected_page == "What's SENTIX":
        home_app()
    elif selected_page == "Review Your Socials":
        page1_app()
    elif selected_page == "Upload Your Data":
        page2_app()
    


