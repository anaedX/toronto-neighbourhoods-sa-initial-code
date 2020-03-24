
import tweepy
import csv

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
mytweet = tweepy.Cursor(api.search, q="(coronavirus OR covid-19) (vancouver) -filter:retweets", lang="en", tweet_mode="extended").items(5000)
cities = ["Vancouver", "vancouver"]
f = open('sentiment_data_vancouver.csv', 'w', encoding='utf-8')
with f:
    for tweet in mytweet:
        txt = tweet.full_text
        match_count = 0;
        for city in cities:
            if txt.find(city) > -1 or txt.find(city.lower()) > -1:
                match_count += 1
        if match_count < 2:
            print(txt+'\n')

            txt_as_row = [txt]
            writer = csv.writer(f)
            writer.writerow(txt_as_row)