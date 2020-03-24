from nltk.corpus import twitter_samples
from sklearn import svm
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import random


if __name__ == "__main__":

    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')

    svm_positive = [[tweet, 1] for tweet in positive_tweets]
    svm_negative = [[tweet, 0] for tweet in negative_tweets]
    svm_dataset = svm_positive + svm_negative
    random.shuffle(svm_dataset)
    dataset_len = round(len(svm_dataset) * 0.7)
    train_vals = [tweet[0] for tweet in svm_dataset]
    train_labels = [tweet[1] for tweet in svm_dataset]

    train_data = train_vals[:dataset_len]
    test_data = train_vals[dataset_len:]
    trained_labels = train_labels[:dataset_len]
    test_labels = train_labels[dataset_len:]
    print(train_data)
    print(trained_labels)

    vectorizer = TfidfVectorizer(min_df=5,
                                 max_df=0.8,
                                 sublinear_tf=True,
                                 use_idf=True)
    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)

    # Perform classification with SVM, kernel=linear
    classifier_linear = svm.SVC(kernel='linear')
    classifier_linear.fit(train_vectors, trained_labels)
    prediction_linear = classifier_linear.predict(test_vectors)

    # results
    report = classification_report(test_labels, prediction_linear, output_dict=True)
    print('report: ', report)

    pickle.dump(vectorizer, open('models/vectorizer.sav', 'wb'))
    pickle.dump(classifier_linear, open('models/svm_classifier.sav', 'wb'))
