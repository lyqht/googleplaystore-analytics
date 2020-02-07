# Google Playstore Analytics 

This repository stores the code for the project for our school course 50.038 Computational Data Science.

## Datasets
- [Google Playstore Data](https://www.kaggle.com/lava18/google-play-store-apps). 
  - There are 2 files, `googleplaystore.csv` and `googleplaystore_user_reviews.csv`.
- [Named Entity Recognition Dataset](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/data#ner_dataset.csv)

# Jupyter Notebooks

We carried out our analysis for the Google App Store Reviews and General App Data in Jupyter Notebooks. 

At the moment, 
- Basic Cleaning for General App Data :`cleaning.ipynb`
- General Visualization of General App Data: `visualization_project.Rmd`
- Preprocessing and Model for Reviews Data using NLTK Naive Bayes Classifier to determine sentiment polarity: `prelim_nlp_model.ipynb` 
- Using preprocessed data from the `prelim_nlp_model.csv`, we vectorize this data and perform cross validation of models for predicting sentiment polarity.
  - `cross_validation.ipynb`
    - MultinomialNB
    - RandomForestClassifier
  - `FastText_Classification.ipynb`: FastTextClassification
  - `SVM.ipynb`: SVM 

Most of the notebooks are well-documented in what they do, you may refer to them for detailed explanation of what they do. The reason for many notebooks is that our group has chosen to work on various tasks *individually*.

# Streamlit App Setup instructions

Install necessary packages
```bash
pip install -r requirements.txt
```

To start the website locally,
```bash
streamlit run website/main.py
```

## Deployment

The Streamlit App is deployed to Heroku, and is redeployed everytime there is a new commit to the master branch.
- The necessary packages are listed in `requirements.txt`
- `setup.sh` is necessary for the dyno to enable the web app server to run on
  -  `enableCORS=false`
  -  `headless=true`
