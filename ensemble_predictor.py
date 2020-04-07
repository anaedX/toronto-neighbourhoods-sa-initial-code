import csv
import pickle
import tweet_cleanup
import random


def test_dataset_by_classifier(datasetName, classifierName):
    f = open('models/'+classifierName, 'rb')
    classifier = pickle.load(f)
    f.close()

    data = tweet_cleanup.clean_tweets('data/'+datasetName, '', False, True)
    print(data)

    i = 0
    sentiment_count = {'Positive': 0, 'Negative': 0}
    for row_text in data:
        if classifierName == 'svm_classifier.sav':
            vectorizer = pickle.load(open('models/svm_vectorizer.sav', 'rb'))
            review_vector = vectorizer.transform([row_text])
            prediction = classifier.predict(review_vector)
            if prediction[0] == 1:
                sentiment_count['Positive'] += 1
            else:
                sentiment_count['Negative'] += 1
            print(i, '---', row_text, prediction)
        elif classifierName == 'nb_classifier.pickle' or classifierName == 'stanford_classifier.pickle':
            custom_tokens = tweet_cleanup.remove_noise(tweet_cleanup.word_tokenize(row_text))
            prediction = classifier.classify(dict([token, True] for token in custom_tokens))
            sentiment_count[prediction] += 1
            print(i, '---', row_text, prediction)
        elif classifierName == 'lr_classifier.sav':
            vectorizer = pickle.load(open('models/lr_vectorizer.sav', 'rb'))
            review_vector = vectorizer.transform([row_text])
            prediction = classifier.predict(review_vector)
            if prediction[0] == 1:
                sentiment_count['Positive'] += 1
            else:
                sentiment_count['Negative'] += 1
            print(i, '---', row_text, prediction)
        elif classifierName == 'rf_classifier.sav':
            vectorizer = pickle.load(open('models/rf_vectorizer.sav', 'rb'))
            review_vector = vectorizer.transform([row_text])
            prediction = classifier.predict(review_vector)
            if prediction[0] == 1:
                sentiment_count['Positive'] += 1
            else:
                sentiment_count['Negative'] += 1
            print(i, '---', row_text, prediction)
        i += 1
    print(sentiment_count)

def ensemble_predict(datasetName, dataArray, dataResults):
    classifiers = ['svm_classifier.sav', 'nb_classifier.pickle', 'lr_classifier.sav', 'rf_classifier.sav']

    # classifier_weights are based on the accuracy score of each classifier divided by the total sum of all classifier accuracy
    classifier_weights = [0.24709137343927357, 0.2642238933030647, 0.24797814982973895, 0.2407065834279228]

    if dataArray:
        data = dataArray
    else:
        data = tweet_cleanup.clean_tweets('data/'+datasetName, '', False, True)

    sentiment_count = {'Positive': 0, 'Negative': 0}
    row_i = 0
    correct_count = 0
    incorrect_count = 0
    for row_text in data:
        positive_count = 0
        negative_count = 0
        c_i = 0
        for classifierName in classifiers:
            f = open('models/' + classifierName, 'rb')
            classifier = pickle.load(f)
            f.close()
            if classifierName == 'svm_classifier.sav':
                vectorizer = pickle.load(open('models/svm_vectorizer.sav', 'rb'))
                review_vector = vectorizer.transform([row_text])
                prediction = classifier.predict(review_vector)
            elif classifierName == 'nb_classifier.pickle' or classifierName == 'stanford_classifier.pickle':
                sentence = [row_text]
                prediction = classifier.predict(sentence)
            elif classifierName == 'lr_classifier.sav':
                vectorizer = pickle.load(open('models/lr_vectorizer.sav', 'rb'))
                review_vector = vectorizer.transform([row_text])
                prediction = classifier.predict(review_vector)
            elif classifierName == 'rf_classifier.sav':
                vectorizer = pickle.load(open('models/rf_vectorizer.sav', 'rb'))
                review_vector = vectorizer.transform([row_text])
                prediction = classifier.predict(review_vector)

            if prediction[0] == 1:
                positive_count += classifier_weights[c_i]
            else:
                negative_count += classifier_weights[c_i]
            c_i += 1

        positive_probability = positive_count/(positive_count+negative_count)
        negative_probability = negative_count/(positive_count+negative_count)

        if(positive_probability > negative_probability):
            sentiment = "Positive"
            sentiment_count['Positive'] += 1
        else:
            sentiment = "Negative"
            sentiment_count['Negative'] += 1

        if(dataResults):
            if((sentiment == "Positive" and dataResults[row_i] == 1) or (sentiment == "Negative" and dataResults[row_i] == 0)):
                correct_count += 1
            else:
                incorrect_count += 1
            print(correct_count, incorrect_count)
        row_i += 1

        print(row_text, sentiment, positive_probability, negative_probability)

    if(dataResults):
        accuracy = correct_count/(correct_count+incorrect_count)
        print('test set accuracy: ', accuracy)
    print(sentiment_count)


# ENSEMBLE CLASSIFER
# ensemble_predict('sentiment_data_toronto.csv')
# ensemble_predict('sentiment_data_montreal.csv')
# ensemble_predict('sentiment_data_vancouver.csv')

# checking training set
dataset = pickle.load(open('data/cleaned_nltk_data.pkl', 'rb'))

random.shuffle(dataset)

train_vals = []
train_labels = []
for tweet in dataset:
    sentence = []
    for word in tweet[0]:
        sentence.append(word)
    train_vals.append(' '.join(sentence))
    if tweet[1] == 'Positive':
        train_labels.append(1)
    else:
        train_labels.append(0)


ensemble_predict('', train_vals, train_labels)


