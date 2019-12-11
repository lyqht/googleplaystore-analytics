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
                - DBOW was selected.
            In DBOW, the paragraph vectors are obtained by training a neural network on the task of predicting a probability distribution of words in a paragraph given a randomly-sampled word from the paragraph.

            After which, we have also performed cross-validation of different models to classify the sentiment of each review.

            ### Cross Validation
            The following are the models that we used:

            1. Decision Tree
                - often perform well on imbalanced datasets due to their hierarchical structure
                - our dataset is biased to positive sentiment
            """)

    st.image("website/assets/DecisionTree.PNG")
    st.write("Figure 1 : Decision Tree")

    st.write("""

            2. Support vector machine
                - This is because we can penalize mistakes on the minority class by an amount proportional to how under-represented it is.
             """)

    st.image("website/assets/SVM.PNG")
    st.write("Figure 2 : SVM")


    st.write("""

            To counter the imbalanced dataset, we also applied SMOTE (Synthetic Minority Over-sampling Technique).
            It is an over-sampling method which creates synthetic samples of the minority class.SMOTE uses a nearest neighbors algorithm to generate new and synthetic data we can use for training our model.

            """)
    #st.image("website/assets/sentiment_freq.jpeg")
    #st.write(""" Figure 3 : Sentiment Distribution""")
    st.write('**10-fold Cross Validation Table**')
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
    st.write(
        """
        Across the apps, there is a differing number of re-views, and the 15th percentile is 28.  
        Thus for train-ing and testing our model, we only selected appsthat minimally have 28 reviews. 
        """
    )
    st.image("website/assets/Predict_explaination.jpg")
    st.write(
        """
        During training, we randomly pick 28 reviews fromeach of these apps and use their sentiment values as nput for the model xi, 
        and the average rating forthe corresponding apps as labels yi.

        Here is the trend curve for 100 epochs.
        """
    )
    st.image("website/assets/RELUgraph.png")
    st.write(
        """
        The model stabiliaed after 100 epoches as there is no major change after that.

    
        """
    )
    
    #Testing running model on the site#

# # Model got weird error when run in my laptop (smth to do with anaconda)
# df = pd.read_csv(
#     "https://raw.githubusercontent.com/lyqht/googleplaystore-analytics/master/data/reviews_joined.csv", index_col=0)
# df.dropna(inplace=True)
# df.reset_index(inplace=True)
# df.drop("index", axis=1, inplace=True)

# num_reviews_per_app = list(df.groupby(["App", "Average_Rating"]).size())
# min_num_reviews = int(np.percentile(num_reviews_per_app, 25))
# num_reviews_per_app = 50  # for input size
# value_counts = df.App.value_counts()
# to_keep = value_counts[value_counts >= num_reviews_per_app].index
# df.drop_duplicates(subset=["Preprocessed_Review",
#                            "App"], inplace=True, keep="first")
# df_dummies = pd.get_dummies(
#     df, prefix_sep='_', drop_first=True, columns=['Category'])

# # instantiate labelencoder object
# ##le = preprocessing.LabelEncoder()
# # apply le on categorical feature columns
# # label the Category in different numbers in the same column
# df_Multi_Category = df
# df_Multi_Category.head()
# df_Multi_Category['Category'].head()
# ##df_Multi_Category['Category'] = le.fit_transform(df_Multi_Category['Category'])

# unique_apps = to_keep

# reviews_by_app = [df[df["App"] == unique_apps[i]]["Preprocessed_Review"].to_numpy(
# ) for i in range(len(to_keep))]  # array contain arrays of reviews of different apps

# review_sentiment_by_app = [df[df["App"] == unique_apps[i]]["Sentiment_Rating"].to_numpy() for i in range(
#     len(to_keep))]  # array containing arrays of reviews' sentiment polarity of different apps

# avr_rating_per_app = [df[df["App"] == unique_apps[i]]["Average_Rating"].to_numpy() for i in range(
#     len(to_keep))]  # array containing the actual average rating of different apps


# class SentimentDataset(Dataset):
#     def __init__(self, x, y):
#         self.samples = x
#         self.labels = y

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         item = self.samples[idx]
#         # sampling 28 reviews from each app
#         item = np.random.choice(item, size=50)
#         return tensor(item, dtype=torch.float), tensor(self.labels[idx][0], dtype=torch.float)


# x = review_sentiment_by_app
# y = avr_rating_per_app

# train_size = int(0.7*len(x))
# val_size = len(x) - train_size

# data = SentimentDataset(review_sentiment_by_app, avr_rating_per_app)
# trainset, valset = random_split(data, [train_size, val_size])
# BATCH_SIZE = 5
# train_dataloader = DataLoader(
#     trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
# val_dataloader = DataLoader(
#     valset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# print("Training dataloader has ", len(
#     train_dataloader), "batches of ", BATCH_SIZE)
# print("Validation dataloader has ", len(
#     val_dataloader), "batches of ", BATCH_SIZE)


# class Net(nn.Module):
#     def __init__(self, n_feature, n_hidden, n_output):
#         super(Net, self).__init__()
#         self.hidden = nn.Linear(n_feature, n_hidden)   # hidden layer
#         self.predict = nn.Linear(n_hidden, n_output)   # output layer

#     def forward(self, x):
#         #xn = F.sigmoid(self.hidden(x))
#         x = F.relu(self.hidden(x))  # activation function for hidden layer
#         x = self.predict(x)             # linear output
#         return x


# INPUT_SIZE = num_reviews_per_app
# OUTPUT_SIZE = 1  # regression to reach average rating
# HIDDEN_SIZE = 30  # arbitrary
# learning_rate = 0.0002

# net = Net(n_feature=INPUT_SIZE, n_hidden=HIDDEN_SIZE, n_output=OUTPUT_SIZE)
# print(net)
# optimizer = optim.Adam(net.parameters(), lr=learning_rate)
# loss_func = nn.MSELoss()
# # writer = SummaryWriter()

# # Trainning
# num_epochs = 100
# losses = []
# for i in range(num_epochs):
#     for batch_idx, samples in enumerate(train_dataloader):
#         x, y = samples

#         prediction = net(x)
#         loss = loss_func(prediction, y)
#         optimizer.zero_grad()   # clear gradients for next train
#         loss.backward()         # backpropagation, compute gradients
#         optimizer.step()        # apply gradients

#         # for plotting
#     if i % 10 == 0:
#         print("Epoch ", i+10, ", Loss: ", loss)
#     losses.append(loss)

# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.plot(range(len(losses)), losses)

# # Testing
# losses = []
# for batch_idx, samples in enumerate(val_dataloader):
#     x, y = samples
#     prediction = net(x)
#     print("Given reviews", x)
#     #print("Actual Average Rating", y)
#     print("Predicted ", prediction)
#     loss = loss_func(prediction, y)
#     print("Loss", loss)
#     losses.append(loss)
#     aver_MSE_loss = sum(losses)/len(losses)
# print(aver_MSE_loss)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.plot(range(len(losses)), losses)

