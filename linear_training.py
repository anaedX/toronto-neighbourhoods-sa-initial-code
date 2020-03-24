from nltk.corpus import twitter_samples
import pickle
from sklearn.linear_model import LogisticRegression
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


if __name__ == "__main__":
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')

    svm_positive = [[tweet, 1] for tweet in positive_tweets]
    svm_negative = [[tweet, 0] for tweet in negative_tweets]
    svm_dataset = svm_positive + svm_negative
    random.shuffle(svm_dataset)
    dataset_len = round(len(svm_dataset) * 0.8)
    train_vals = [tweet[0] for tweet in svm_dataset]
    train_labels = [tweet[1] for tweet in svm_dataset]

    train_data = train_vals[:dataset_len]
    test_data = train_vals[dataset_len:]
    trained_labels = train_labels[:dataset_len]
    test_labels = train_labels[dataset_len:]

    vectorizer = TfidfVectorizer(min_df=5,
                                 max_df=0.8,
                                 sublinear_tf=True,
                                 use_idf=True)
    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)
    print(train_vectors)
    print(trained_labels)

    regressor = LogisticRegression()
    regressor.fit(train_vectors, trained_labels)
    # To retrieve the intercept:
    print(regressor.intercept_)
    # For retrieving the slope:
    print(regressor.coef_)
    y_pred = regressor.predict(test_vectors)
    score = regressor.score(test_vectors, test_labels)
    print('score : ', score)

    pickle.dump(vectorizer, open('models/vectorizer_lr.sav', 'wb'))
    pickle.dump(regressor, open('models/lr_classifier.sav', 'wb'))
