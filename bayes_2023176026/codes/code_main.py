import pdb
from nltk.corpus import stopwords, twitter_samples
import numpy as np
import pandas as pd
import nltk
import string
from nltk.tokenize import TweetTokenizer
from os import getcwd
from bayes_2023176026.codes.error_analysis import CheckMisclassifiedTweets
from bayes_2023176026.codes.filter_words import print_words_by_threshold
from bayes_2023176026.codes.predict import NaiveBayesClassifierPredictor
from bayes_2023176026.codes.test import NaiveBayesTester
from bayes_2023176026.codes.train import NaiveBayesClassifierTrainer

from bayes_2023176026.codes.utility_functions import *;

nltk.download('twitter_samples')

filePath = f"{getcwd()}/../tmp2/"
nltk.data.path.append(filePath)

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')


test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))

nltk.download('stopwords')
freqs = utility_functions.count_tweets({}, train_x, train_y)

#***--- Training ---***
print("-"*20)
print("Training:")
logprior, loglikelihood = NaiveBayesClassifierTrainer().train(freqs, train_x, train_y)
print("logprior: ",logprior)
print("Total values in loglikelihood: ",len(loglikelihood))


#***--- Predict ---***
print("-"*20)
print("Predictions:")
for tweet in ['I am happy because leaning NLP','I am sad not leaning NLP']:
    p = NaiveBayesClassifierPredictor(logprior, loglikelihood).predict(tweet)
    print(f'{tweet} -> {p:.2f} ')
    if p > 0:
        print('Positive sentiment')
    else:
        print('Negative sentiment')


#***--- Accuracy ---***
print("-"*20)
print("Naive Bayes accuracy = %0.4f" %(NaiveBayesTester(logprior, loglikelihood).test_naive_bayes(test_x, test_y)))

#***---  Filter words by Ratio of positive to negative counts ---***
print("-"*20)
print("\nNegative words at or below a threshold:")
print_words_by_threshold(freqs, label=0, threshold=0.05)
print("\nPositive words at or above a threshold:")
print_words_by_threshold(freqs, label=1, threshold=10)

#***--- Error Analysis ---***
print("-"*20)
print("Check misclassified tweets:")

misclassified_tweets = CheckMisclassifiedTweets(logprior, loglikelihood).check_misclassified_tweets(test_x, test_y)
print("Number of misclassified tweets: ", len(misclassified_tweets))