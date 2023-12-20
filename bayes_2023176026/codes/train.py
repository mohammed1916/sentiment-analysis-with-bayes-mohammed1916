import numpy as np

from utility_functions import utility_functions 

class NaiveBayesClassifierTrainer:
    def __init__(self):
        self.loglikelihood = {}
        self.logprior = 0

    def train(self, freqs, train_x, train_y):
        '''
        Input:
            freqs: dictionary from (word, label) to how often the word appears
            train_x: a list of tweets
            train_y: a list of labels correponding to the tweets (0,1)
        Output:
            logprior: the log prior. log(Dpos)-log(Dneg)
            loglikelihood: the log likelihood of Naive bayes equation. log(P(W_pos)/P(W_neg)) where P(W_pos)=freq_pos+1/N_pos+V and P(W_neg)=freq_neg+1/N_neg+V
        '''
        vocab = set([pair[0] for pair in freqs.keys()])
        V = len(vocab)

        N_pos = N_neg = 0
        for pair in freqs.keys():
            if pair[1] > 0:
                N_pos += freqs[pair]
            else:
                N_neg += freqs[pair]

        D = len(train_y)
        print("Total Number of Documents:", D)

        D_pos = np.sum(train_y)
        print("Number of positive documents:", D_pos)

        D_neg = D - D_pos
        print("Number of negative documents:", D_neg)

        self.logprior = np.log(D_pos) - np.log(D_neg)

        for word in vocab:
            freq_pos = utility_functions.lookup(freqs, word, 1)
            freq_neg = utility_functions.lookup(freqs, word, 0)

            p_w_pos = (freq_pos + 1) / (N_pos + V)
            p_w_neg = (freq_neg + 1) / (N_neg + V)

            self.loglikelihood[word] = np.log(p_w_pos / p_w_neg)

        return self.logprior, self.loglikelihood
