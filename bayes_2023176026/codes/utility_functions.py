import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

class utility_functions:
    def process_tweet(tweet):
        '''
        Input:
            tweet: a string containing a tweet
        Output:
            tweets_clean: a list of words containing the processed tweet

        '''
        stemmer = PorterStemmer()
        stopwords_english = stopwords.words('english')
        tweet = re.sub(r'\$\w*', '', tweet)
        tweet = re.sub(r'^RT[\s]+', '', tweet)
        tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
        tweet = re.sub(r'#', '', tweet)
        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
        tweet_tokens = tokenizer.tokenize(tweet)

        tweets_clean = []
        for word in tweet_tokens:
            if (word not in stopwords_english and word not in string.punctuation): 
                stem_word = stemmer.stem(word) 
                tweets_clean.append(stem_word)

        return tweets_clean


    def lookup(freqs, word, label):
        '''
        Input:
            freqs: a dictionary with the frequency of each pair (or tuple)
            word: the word to look up
            label: the label corresponding to the word
        Output:
            n: the number of times the word with its corresponding label appears.
        '''
        n = 0  # freqs.get((word, label), 0)

        pair = (word, label)
        if (pair in freqs):
            n = freqs[pair]

        return n


    def count_tweets(result, tweets, ys):
        '''
        Input:
            result: a dictionary that will be used to map each pair to its frequency
            tweets: a list of tweets
            ys: a list corresponding to the sentiment of each tweet (either 0 or 1)
        Output:
            result: a dictionary mapping each pair to its frequency
        '''

        for y, tweet in zip(ys, tweets):
            for word in utility_functions.process_tweet(tweet):
                pair = (word,y)
                if pair in result:
                    result[pair] += 1
                else:
                    result[pair] = 1
        return result