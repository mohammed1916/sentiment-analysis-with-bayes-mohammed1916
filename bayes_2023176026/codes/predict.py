from bayes_2023176026.codes.utility_functions import utility_functions

class NaiveBayesClassifierPredictor:
    def __init__(self, logprior, loglikelihood):
        self.logprior = logprior
        self.loglikelihood = loglikelihood

    def predict(self, tweet):
        '''
        Input:
            tweet: a string
        Output:
            p: the sum of all the loglikelihoods of each word in the tweet (if found in the dictionary) + logprior (a number)
        '''
        word_l = utility_functions.process_tweet(tweet)
        p = 0
        p += self.logprior

        for word in word_l:
            if word in self.loglikelihood:
                p += self.loglikelihood[word]
        return p
