import numpy as np
from bayes_2023176026.codes.predict import NaiveBayesClassifierPredictor
from bayes_2023176026.codes.utility_functions import utility_functions


class CheckMisclassifiedTweets:
    def __init__(self, logprior, loglikelihood):
        self.logprior = logprior
        self.loglikelihood = loglikelihood

    def check_misclassified_tweets(self,test_x, test_y):
        misclassified_tweets = []
        for x, y in zip(test_x, test_y):
            y_hat = NaiveBayesClassifierPredictor(self.logprior, self.loglikelihood).predict(x)
            if np.abs(y - (y_hat > 0)) > 0:
                misclassified_tweets.append((x, y, y_hat))
                print('%d\t%0.2f\t%s' % (y, np.sign(y_hat) > 0, ' '.join(utility_functions.process_tweet(x)).encode('ascii', 'ignore')))

        return misclassified_tweets