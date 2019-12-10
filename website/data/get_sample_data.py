import requests
import json
import pandas as pd

json_path = "sample_data.json"
store = "android"       # Could be either "android" or "itunes".
language = "en"         # Two letter language code.
req_params = {"language": language}
# request for free api key from appmonsta
username = "YOUR_API_USERNAME"
password = "X"          # Password can be anything.


def appmonsta_retrieve_reviews():
    url = "https://api.appmonsta.com/v1/stores/%s/reviews.json" % store
    headers = {'Accept-Encoding': 'deflate, gzip'}
    response = requests.get(url,
                            auth=(username, password),
                            headers=headers,
                            params=req_params,
                            stream=True)
    print("Got Response!")

    reviews = []
    # now write output to a file
    for line in response.iter_lines():
        line = json.loads(line)
        rating = line["rating"]
        app_id = line["app_id"]
        date = line["date"]
        review_text = ""
        if "title" in line:
            if type(line["title"]) == str:
                review_text += line["title"]
        if "review_text" in line:
            if type(line["review_text"]) == str:
                review_text += line["review_text"]
        reviews.append([review_text, rating, app_id, date])
    return reviews


def get_app_id():
    url = "https://api.appmonsta.com/v1/stores/%s/reviews.json" % store
    headers = {'Accept-Encoding': 'deflate, gzip'}
    response = requests.get(url,
                            auth=(username, password),
                            headers=headers,
                            params=req_params,
                            stream=True)
    print("Got Response!")

    reviews = []
    # now write output to a file
    for line in response.iter_lines():
        line = json.loads(line)
        rating = line["rating"]
        app_id = line["app_id"]
        date = line["date"]
        review_text = ""
        if "title" in line:
            if type(line["title"]) == str:
                review_text += line["title"]
        if "review_text" in line:
            if type(line["review_text"]) == str:
                review_text += line["review_text"]
        reviews.append([review_text, rating, app_id, date])
    return reviews


def get_df(reviews):
    df = pd.DataFrame(reviews)
    df.columns = ["Review Text", "Rating", "App id", "Date"]
    df.to_csv("sample_reviews.csv")


reviews = appmonsta_retrieve_reviews()
get_df(reviews)
