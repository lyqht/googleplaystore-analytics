import streamlit as st
import numpy as np
import pandas as pd
import time

st.title("Sentiment Analysis on Google Playstore Reviews")


@st.cache
def load_data():
    reviews_data = pd.read_csv("data/googleplaystore_user_reviews.csv")
    app_data = pd.read_csv("data/googleplaystore.csv")
    return reviews_data, app_data


data_load_state = st.text("Loading data...")
reviews_df, app_df = load_data()
data_load_state.text("Loading data... done!")

"General App Data"
app_df

# Sidebar stuff
# columns = df.columns
# for option in df.columns:
#     st.sidebar.selectbox(option, [False, True])
