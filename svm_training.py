from nltk.corpus import twitter_samples
from sklearn import svm
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import random
from sklearn.model_selection import GridSearchCV



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

    # Perform classification with SVM - hypertuning used
    classifier_linear = svm.SVC(C=1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf', max_iter=-1,
    probability=False, random_state=None, shrinking=True, tol=0.001,
    verbose=False)
    classifier_linear.fit(train_vectors, trained_labels)
    prediction_linear = classifier_linear.predict(test_vectors)
    # results
    report = classification_report(test_labels, prediction_linear, output_dict=True)
    total += report['accuracy']
    print('report: ', report)

#hyper tuning
# defining parameter range
# param_grid = {'C': [0.1, 1, 10, 100, 1000],
#               'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#               'kernel': ['rbf']}
# grid = GridSearchCV(svm.SVC(kernel='linear'), param_grid, refit=True, verbose=3)
# # fitting the model for grid search
# grid.fit(train_vectors, trained_labels)
# # print best parameter after tuning
# print(grid.best_params_)
# # print how our model looks after hyper-parameter tuning
# print(grid.best_estimator_)
# grid_predictions = grid.predict(test_vectors)
# # print classification report
# print(classification_report(test_labels, grid_predictions))

average_accuracy = total/num_folds;
print('average accuracy is : ', average_accuracy)

#
pickle.dump(vectorizer, open('models/svm_vectorizer.sav', 'wb'))
pickle.dump(classifier_linear, open('models/svm_classifier.sav', 'wb'))
