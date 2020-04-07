import pickle
import random
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report


dataset = pickle.load(open('data/cleaned_nltk_data.pkl', 'rb'))

random.shuffle(dataset)

num_folds = 10
subset_size = round(len(dataset)/num_folds)
total = 0

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

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])
tuned_parameters = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': [1, 1e-1, 1e-2]
}

x_train, x_test, y_train, y_test = train_test_split(train_vals, train_labels, test_size=0.2, random_state=42)
clf = GridSearchCV(text_clf, tuned_parameters, cv=10, scoring='accuracy')
clf.fit(x_train, y_train)

print(classification_report(y_test, clf.predict(x_test), digits=4))

print(clf.best_params_)

f = open('models/nb_classifier.pickle', 'wb')
pickle.dump(clf, f)
f.close()
