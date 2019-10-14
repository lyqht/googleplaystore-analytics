import streamlit as st
import numpy as np
import pandas as pd
import time


st.title("Google App Analytics")
st.write("## Sentiment Analysis on Reviews")


@st.cache
def load_data():
    data = pd.read_csv("data/googleplaystore_user_reviews.csv")
    return data


data_load_state = st.text("Loading data...")
df = load_data()
data_load_state.text("Loading data... done!")

st.write("Here's a sample of the reviews' data.")
df.head(10)


# Sidebar stuff
columns = df.columns
for option in df.columns:
    st.sidebar.selectbox(option, [False, True])
