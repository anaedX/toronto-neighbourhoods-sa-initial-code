import csv
import json

positive_tweets = []
negative_tweets = []
rows_to_read = 20001

with open('stanford_data.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == '0' and len(negative_tweets) < rows_to_read:
               negative_tweets.append(row[5])
            elif row[0] == '4' and len(positive_tweets) < rows_to_read:
               positive_tweets.append(row[5])


with open('stanford_positive_tweets.json', 'w') as outfile:
    json.dump(positive_tweets, outfile)

with open('stanford_negative_tweets.json', 'w') as outfile:
    json.dump(negative_tweets, outfile)