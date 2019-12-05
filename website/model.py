import streamlit as st


def write():
    st.title("Prediction of Individual Star Rating")

    st.write("""
             ## Feature Engineering: Sentiment Polarity
                          
            Since we **do not have individual star ratings for each review**, 
            [In the first checkoff](https://docs.google.com/presentation/d/e/2PACX-1vSopfa2P6Pq1XkO0fVyysrAUlShHuYO1YM0bXarXPL4majGsw1EUvd0gxvwepYRxl89yiJfglcmTmdH/pub?start=true&loop=false&delayms=3000), 
            we generated labels using VADER SentimentIntensityAnalyzer and used NaiveBayesClassifier to classify the reviews based on the sentiment polarity.
            
            
            After which, we have also performed cross-validation of different models to predict the sentiment polarity of each reviews.
             """)

    # insert table here

    st.write("After finalizing the classifier model for predicting the sentiment polarity, we designed a Neural Network with Pytorch as such.")
