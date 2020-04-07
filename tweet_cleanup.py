from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import json
import emoji
import csv
import pickle


import re, string, random


def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = emoji.demojize(token, delimiters=("",""))
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)
        token = re.sub(r'[^\x00-\x7F]+', '', token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


def clean_tweets(dataset, label, is_json, return_sentences):
    if is_json:
        with open(dataset) as json_file:
            tweets = json.load(json_file)
            tweet_tokens = []
            for tweet in tweets:
                tweet = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                   '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', tweet)
                tweet_tokens.append(word_tokenize(tweet))
    else:
        with open(dataset, 'r', encoding='utf-8') as file:
            tweets = csv.reader(file)
            tweet_tokens = []
            for row in tweets:
                row_text = '\n'.join(row)
                if (len(row_text) > 0):
                    row_text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', row_text)
                    tweet_tokens.append(word_tokenize(row_text))

    stop_words = stopwords.words('english')

    cleaned_tokens_list = []

    for tokens in tweet_tokens:
        cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    tokens_for_model = get_tweets_for_model(cleaned_tokens_list)

    dataset = [(tweet_dict, label)
               for tweet_dict in tokens_for_model]
    if return_sentences:
        train_vals = []
        for tweet in dataset:
            sentence = []
            for word in tweet[0]:
                sentence.append(word)
            train_vals.append(' '.join(sentence))
        return train_vals
    else:
        return dataset


# positive_tweets_cleaned = clean_tweets('data/stanford_positive_tweets.json', "Positive", True, False)
# negative_tweets_cleaned = clean_tweets('data/stanford_negative_tweets.json', "Negative", True, False)
# combined_dataset = positive_tweets_cleaned + negative_tweets_cleaned
#
# with open('data/cleaned_nltk_data.pkl', 'wb') as f:
#     pickle.dump(combined_dataset, f)