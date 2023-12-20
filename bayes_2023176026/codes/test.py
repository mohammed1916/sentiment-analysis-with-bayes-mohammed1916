import numpy as np
from bayes_2023176026.codes.predict import NaiveBayesClassifierPredictor;

class NaiveBayesTester:
    def __init__(self, logprior, loglikelihood):
        self.logprior = logprior
        self.loglikelihood = loglikelihood

    def test_naive_bayes(self, test_x, test_y):
        """
        Input:
            test_x: A list of tweets
            test_y: the corresponding labels for the list of tweets
        Output:
            accuracy: (# of tweets classified correctly)/(total # of tweets)
        """
        accuracy = 0
        y_hats = []
        
        for tweet in test_x:
            if NaiveBayesClassifierPredictor(self.logprior, self.loglikelihood).predict(tweet) > 0:
                y_hat_i = 1
            else:
                y_hat_i = 0
            y_hats.append(y_hat_i)

        error = np.mean(np.absolute(y_hats-test_y))
        accuracy = 1-error
        return accuracy
