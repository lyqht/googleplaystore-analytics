# Google Playstore Analytics 

This repository stores the code for the project for our school course 50.038 Computational Data Science.

The dataset is retrieved from Kaggle [here](https://www.kaggle.com/lava18/google-play-store-apps). There are 2 files, `googleplaystore.csv` and `googleplaystore_user_reviews.csv`.

# Jupyter Notebooks

We carried out our analysis for the Google App Store Reviews and General App Data in Jupyter Notebooks. 

At the moment, what we have done:
- `cleaning.ipynb`: Cleaning for General App Data 
- `visualization_project.Rmd`: General Visualization of General App Data
- `prelim_nlp_model.ipynb`: Preprocessing and Model for Reviews Data using NLTK Naive Bayes Classifier to determine sentiment polarity
- `cross_validation.ipynb`: Using preprocessed data from the `prelim_nlp_model.csv`, we vectorize this data and perform cross validation of models for predicting sentiment polarity.
  - MultinomialNB
  - RandomForestClassifier


Most of the notebooks are well-documented in what they do, you may refer to them for detailed explanation of what they do. The reason for many notebooks is that our group has chosen to work on various tasks individually.

# Streamlit App Setup instructions

Install necessary packages
```bash
pip install -r requirements.txt
```

To start the website locally,
```bash
streamlit run website/main.py
```

At the moment, it only shows a preview of the reviews' data. We are still planning what to actually show on this app for interaction.

## Deployment

The Streamlit App is deployed to Heroku, and is redeployed everytime there is a new commit to the master branch.
- The necessary packages are listed in `requirements.txt`
- `setup.sh` is necessary for the dyno to enable the web app server to run on
  -  `enableCORS=false`
  -  `headless=true`