import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


@st.cache
def load_data():
    reviews_data = pd.read_csv("data/reviews_naive_polarity.csv")
    app_data = pd.read_csv("data/googleplaystore_cleaned.csv")
    return reviews_data, app_data


view_modes = ["General App Data", "Reviews Data"]


def preview(selection, reviews_df, app_df):
    if selection == view_modes[0]:
        preview_general(app_df)
    else:
        preview_reviews(reviews_df)


def preview_general(df):
    st.write("General App Data")
    st.write(df)

    st.write("""
             ## Observation of Trends
             """)
    fig = plt.figure(figsize=(10, 4))
    plt.title('Frequency Distribution of Categories')
    plt.xlabel('Category')
    plt.ylabel('Frequency')
    sns.countplot(df['Category'])
    plt.xticks(rotation=90)
    plt.tight_layout()
    st.pyplot()

    fig = plt.figure(figsize=(10, 4))
    sns.countplot(df['Category'])
    plt.xlabel('Genre')
    plt.ylabel('Frequency')
    plt.title('Frequency Distribution of Genres')
    plt.xticks(rotation=90)
    plt.tight_layout()
    st.pyplot()

    fig = plt.figure(figsize=(10, 4))
    sns.countplot(df['Content Rating'])
    plt.xlabel('Content Rating')
    plt.ylabel('Frequency')
    plt.title('Frequency Distribution of Content Ratings')
    plt.tight_layout()
    st.pyplot()

    fig = plt.figure(figsize=(10, 4))
    sns.countplot(df['Type'], orient="h")
    plt.title('Frequency of Free and Paid Apps')
    st.pyplot()

    fig = plt.figure(figsize=(10, 4))
    sns.countplot(df['Installs'])
    plt.title('Frequency of Install Count')
    plt.xticks(rotation=90)
    st.pyplot()

    fig = plt.figure(figsize=(10, 4))
    sns.distplot(df['Size'].dropna())
    plt.title('Frequency of App size')
    st.pyplot()

    fig = plt.figure(figsize=(10, 4))
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    sns.countplot(df['Rating'])
    plt.title('Frequency of Average Star Ratings')
    st.pyplot()


def preview_reviews(df):
    st.write("Reviews Data")
    st.write(df)


def write():
    reviews_df, app_df = load_data()
    st.write(
        """
        ### Skimming Through The Dataset
        We have 2 datasets, one containing general app data, and the other containing reviews.
        The data shown here has been preprocessed already.
        """)
    selection = st.radio("Explore: ", view_modes)
    preview(selection, reviews_df, app_df)
