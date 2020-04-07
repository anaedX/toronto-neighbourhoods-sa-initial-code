from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import random
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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
    trained_labels = train_labels[i * subset_size:][:subset_size]

    vectorizer = TfidfVectorizer(min_df=5,
                                 max_df=0.8,
                                 sublinear_tf=True,
                                 use_idf=True)

    train_vectors = vectorizer.fit_transform(train_data).toarray()

    X_train, X_test, y_train, y_test = train_test_split(train_vectors, trained_labels, test_size=0.2, random_state=0)

    text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
    text_classifier.fit(X_train, y_train)

    predictions = text_classifier.predict(X_test)
    total += accuracy_score(y_test, predictions)
    print(accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions, digits=4))

average_accuracy = total/num_folds;
print('average accuracy is : ', average_accuracy)

pickle.dump(vectorizer, open('models/rf_vectorizer.sav', 'wb'))
pickle.dump(text_classifier, open('models/rf_classifier.sav', 'wb'))
