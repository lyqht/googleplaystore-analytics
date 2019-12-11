import streamlit as st


def write():
    st.title("Prediction of Individual Star Rating")
    st.write("""
             ## Feature Engineering: Sentiment Polarity

            Since we **do not have individual star ratings for each review**,
            we generated labels using VADER SentimentIntensityAnalyzer and used NaiveBayesClassifier to classify the reviews based on the sentiment polarity.
            [(link to previous presentation slides on this)](https://docs.google.com/presentation/d/e/2PACX-1vSopfa2P6Pq1XkO0fVyysrAUlShHuYO1YM0bXarXPL4majGsw1EUvd0gxvwepYRxl89yiJfglcmTmdH/pub?start=true&loop=false&delayms=3000)

            ## Classification Model
            After we labelled the reviews with their corresponding sentiment, we proceeded to build a classification model for classifying the reviews into their corresponding sentiment.
            For the input of the classifier, we explored two different feature extraction method for the reviews:

            1. TF-IDF
            2. Doc2Vec
                - For Doc2Vec, there are distributed bag of words (DBOW) and distributed memory (DM).
                - However, we chose ***DBOW*** as it is known to be able to perform better for a smaller corpus.
              In DBOW, the paragraph vectors are obtained by training a neural network on the task of predicting a probability distribution of words in a paragraph given a randomly-sampled word from the paragraph.

            After which, we have also performed cross-validation of different models to classify the sentiment of each review.

            ### Cross Validation
            The following are the models that we used:

            1. Decision Tree
                - often perform well on imbalanced datasets as their hierarchical structure allows them to learn signals from both classes.
                - our dataset is biased to positive sentiment
            """)

    st.image("website/assets/DecisionTree.PNG",use_column_width=True)
    st.write("Figure 1 : Decision Tree")

    st.write("""

            2. Support vector machine
                - This is because we can penalize mistakes on the minority class by an amount proportional to how under-represented it is by using the argument class weight = ‘balanced’.
             """)

    st.image("website/assets/SVM.PNG",use_column_width=True)
    st.write("Figure 2 : SVM")


    st.write("""

            To counter the imbalanced dataset, we also applied SMOTE (Synthetic Minority Over-sampling Technique).
            It is an over-sampling method which creates synthetic samples of the minority class.SMOTE uses a nearest neighbors algorithm to generate new and synthetic data we can use for training our model.
            We use `imblearn` python package to achieve this.
            """)
    #st.image("website/assets/sentiment_freq.png")
    st.write(""" Figure 3 : Sentiment Distribution

            **10-fold Cross Validation Table**
             """)
    st.image("website/assets/Cross_Validation.PNG")
    st.write("Figure 4 : Cross Validation")
    st.write("""
             Using TF-IDF generally results in higher F1 score.
             """)

    st.write("""
            The measure of performance is micro-average F1-score as the dataset is unbalanced.
            The following is our best model on classification.

            **TF-IDF** + **Random Forest** + **OverSampling**

             """)
    st.image("website/assets/Best_Model.PNG")
    st.write("Figure 5 : Best Model")


    st.write("""
             ## Individual Star Rating Prediction Model
             After finalizing the classifier model for predicting the sentiment polarity, we designed a Neural Network with `Pytorch` as such.
             """)

    # TODO: insert image on general idea of the model
    st.image("website/assets/ReLUgraph.png")
    st.write("Figure 6 : ReLUgraph")
