import streamlit as st

import dataset
import intro
import model
import user_prediction
import references

PAGES = {
    "Intro": intro,
    "Dataset Exploration": dataset,
    "Modelling": model,
    "Prediction Based on User Input": user_prediction,
    "References": references
}


def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]

    with st.spinner(f"Loading Page ..."):
        page.write()  # each page has a write function


if __name__ == "__main__":
    main()
