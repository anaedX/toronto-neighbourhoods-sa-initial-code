import preprocessor as p
import csv
import emoji
import nltk
import heapq
import svm_training
import pickle
import nb_training
from sklearn.feature_extraction.text import TfidfVectorizer

def test_dataset(datasetName, classifierName):
    f = open('models/'+classifierName, 'rb')
    classifier = pickle.load(f)
    f.close()

    with open('data/'+datasetName, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        i = 0
        sentiment_count = {'Positive': 0, 'Negative': 0}
        for row in reader:
            row_text = '\n'.join(row)
            if (len(row_text) > 0):
                if classifierName == 'svm_classifier.sav':
                    vectorizer = pickle.load(open('models/vectorizer.sav', 'rb'))
                    review_vector = vectorizer.transform([row_text])
                    prediction = classifier.predict(review_vector)
                    if prediction[0] == 1:
                        sentiment_count['Positive'] += 1
                    else:
                        sentiment_count['Negative'] += 1
                    print(i, '---', row_text, prediction)
                elif classifierName == 'nb_classifier.pickle' or classifierName == 'stanford_classifier.pickle':
                    custom_tokens = nb_training.remove_noise(nb_training.word_tokenize(row_text))
                    prediction = classifier.classify(dict([token, True] for token in custom_tokens))
                    sentiment_count[prediction] += 1
                    print(i, '---', row_text, prediction)
                if classifierName == 'lr_classifier.sav':
                    vectorizer = pickle.load(open('models/vectorizer_lr.sav', 'rb'))
                    review_vector = vectorizer.transform([row_text])
                    prediction = classifier.predict(review_vector)
                    if prediction[0] == 1:
                        sentiment_count['Positive'] += 1
                    else:
                        sentiment_count['Negative'] += 1
                    print(i, '---', row_text, prediction)
                i += 1
        print(sentiment_count)

#SVM
# test_dataset('sentiment_data_toronto.csv', 'svm_classifier.sav')
# test_dataset('sentiment_data_montreal.csv', 'svm_classifier.sav')
# test_dataset('sentiment_data_vancouver.csv', 'svm_classifier.sav')

#NB
# test_dataset('sentiment_data_toronto.csv', 'nb_classifier.pickle')
# test_dataset('sentiment_data_montreal.csv', 'nb_classifier.pickle')
# test_dataset('sentiment_data_vancouver.csv', 'nb_classifier.pickle')

#Stanford NB
# test_dataset('sentiment_data_toronto.csv', 'stanford_classifier.pickle')
# test_dataset('sentiment_data_montreal.csv', 'stanford_classifier.pickle')
test_dataset('sentiment_data_vancouver.csv', 'stanford_classifier.pickle')

#LINEAR
# test_dataset('sentiment_data_toronto.csv', 'lr_classifier.sav')
# test_dataset('sentiment_data_montreal.csv', 'lr_classifier.sav')
# test_dataset('sentiment_data_vancouver.csv', 'lr_classifier.sav')

