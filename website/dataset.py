import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


@st.cache
def load_data():
    reviews_data = pd.read_csv("data/reviews_naive_polarity.csv")
    app_data = pd.read_csv("data/googleplaystore_cleaned.csv")
    ner_data = pd.read_csv("data/ner_dataset.csv", encoding="latin1")
    return reviews_data, app_data, ner_data


view_modes = ["General App Data", "Reviews Data", "NER"]


def preview(selection, reviews_df, app_df, ner_df):
    if selection == view_modes[0]:
        preview_general(app_df)
    elif selection == view_modes[1]:
        preview_reviews(reviews_df)
    else:
        preview_ner(ner_df)


def preview_general(df):
    st.write("## General App Data")
    st.write(df)

    st.write("""
             ### Observation of Trends
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
    # sns.countplot(df['Genres'])
    value_counts = df["Genres"].value_counts()
    genre_labels = list(value_counts.index)

    # manually overriding so that they don't appear on the circle
    for i in range(10, len(genre_labels)):
        genre_labels[i] = " "

    def my_autopct(pct):
        return ('%.2f%%' % pct) if pct > 3.5 else ''

    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    fig1, ax1 = plt.subplots()
    ax1.pie(value_counts, colors=colors, labels=genre_labels,
            autopct=my_autopct, startangle=90)

    # wanted to add in an explode param, but streamlit did not allow this
    # https://medium.com/@kvnamipara/a-better-visualisation-of-pie-charts-by-matplotlib-935b7667d77f

    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    ax1.axis('equal')
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
    st.write("## Reviews Data")
    st.write(df)


def preview_ner(df):
    st.write("## Ner Data")
    st.write(df)


def write():
    reviews_df, app_df, ner_df = load_data()
    st.write(
        """
        ### Skimming Through The Dataset
        Toggle below to view the different datasets! (already been preprocessed by us) 
        """)
    selection = st.radio("Explore: ", view_modes)
    preview(selection, reviews_df, app_df, ner_df)
