import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import itertools
from nltk.probability import FreqDist


@st.cache(persist=True)
def load_data():
    reviews_data = pd.read_csv("data/reviews_naive_polarity.csv")
    app_data = pd.read_csv("data/googleplaystore_cleaned.csv")
    return reviews_data, app_data


view_modes = ["General App Data", "Reviews Data"]


def preview(selection, reviews_df, app_df):
    if selection == view_modes[0]:
        preview_general(app_df)
    elif selection == view_modes[1]:
        preview_reviews(reviews_df)


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
    st.write(
        "## Reviews Data\n The preprocessing steps taken to produce the tokens can refered [here](https://github.com/lyqht/googleplaystore-analytics/blob/master/Notebooks/prelim_nlp_model.ipynb) ")
    st.write(df)

    st.write("Word Count Frequency")
    st.image("website/assets/freqDist.png", use_column_width=True)
    st.write("""### Topic Modelling""")
    st.write("""
             Refer [here](website/assets/overall_100_topics_enhanced.html) for pyLDAvis visualization. 
             Topic modelling is based on the Latent Dirichlet Allocation Algorithm.
             """)

    def show_lda_explanation():
        st.write("#### Latent Dirichlet Algorithm (LDA)")
        st.write(r"""
            To begin, LDA is based on the Dirichlet Distribution, normally known as $Dir(\alpha)$.
            Dirichlet distributions are commonly used as prior distributions in Bayesian statistics, 
            and in fact the Dirichlet distribution is the conjugate prior of the categorical distribution and multinomial distribution.
            A great article about this distribution can be found [here](https://towardsdatascience.com/dirichlet-distribution-a82ab942a879).
            """)
        st.image("website/assets/Dirichlet.png", use_column_width=True)
        st.write(r"""
        Given:
        - A document is a sequence of $N$ words denoted by $\textbf{w} = (w_1,w_2,... ,w_N)$, where $w_n$ is the nth word b in the sequence.
        - A corpus is a collection of $M$ documents denoted by $D = \textbf{w}_1, \textbf{w}_2,...\textbf{w}_m$
        - $\alpha$ is the Dirichlet prior on the per-document topic distributions
        - $\beta$ is the Dirichlet prior on the per-topic  word distributions
        - $\Theta$ is the topic distribution for document $m$
        - $z_{mn}$ is the topic for $n^{\text{th}}$  word in document $m$
        """)
        st.image("website/assets/LDA-concept2.png", use_column_width=True)
        st.image("website/assets/LDA-concept.png", use_column_width=True)

    lda_explanation = st.checkbox("Show explanation on LDA")
    if lda_explanation:
        show_lda_explanation()


def write():
    reviews_df, app_df = load_data()
    st.write(
        """
        ### Skimming Through The Dataset
        Toggle below to view the different datasets! (already been preprocessed by us) 
        """)
    selection = st.radio("Explore: ", view_modes)
    preview(selection, reviews_df, app_df)
