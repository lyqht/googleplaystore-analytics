# Google Playstore Analytics 

This repository stores the code for the project for our school course 50.038 Computational Data Science.

The dataset is retrieved from Kaggle [here](https://www.kaggle.com/lava18/google-play-store-apps) .

# Jupyter Notebooks

We carried out our analysis for the Google App Store Reviews and general App data in Jupyter Notebooks first. 

At the moment, we have done:
- General Visualization of Google Playstore general app data in `googleplaystore.csv`

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