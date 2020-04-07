from nltk.corpus import twitter_samples
import pickle
from sklearn.linear_model import LogisticRegression
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report



dataset = pickle.load(open('data/cleaned_nltk_data.pkl', 'rb'))

random.shuffle(dataset)
dataset_len = round(len(dataset) * 0.9)
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

num_folds = 10
subset_size = round(len(dataset)/num_folds)
total = 0
# using k-fold cross validation
for i in range(num_folds):
    train_data = train_vals[i * subset_size:][:subset_size]
    test_data = train_vals[:i * subset_size] + train_vals[(i + 1) * subset_size:]
    trained_labels = train_labels[i * subset_size:][:subset_size]
    test_labels = train_labels[:i * subset_size] + train_labels[(i + 1) * subset_size:]

    vectorizer = TfidfVectorizer(min_df=5,
                                 max_df=0.8,
                                 sublinear_tf=True,
                                 use_idf=True)
    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)

    # # results
    regressor = LogisticRegression(C=1.0, penalty='l2')
    regressor.fit(train_vectors, trained_labels)

    y_pred = regressor.predict(test_vectors)
    score = regressor.score(test_vectors, test_labels)
    total += score
    print(classification_report(test_labels, regressor.predict(test_vectors), digits=4))

average_accuracy = total/num_folds;
print('average accuracy is : ', average_accuracy)


pickle.dump(vectorizer, open('models/lr_vectorizer.sav', 'wb'))
pickle.dump(regressor, open('models/lr_classifier.sav', 'wb'))
