from prettytable import PrettyTable
from bayes_2023176026.codes.utility_functions import utility_functions


def get_ratio(freqs, word):
    '''
    Input:
        freqs: dictionary containing the words
        word: string to lookup

    Output: a dictionary with keys 'positive', 'negative', and 'ratio'.
        Example: {'positive': 10, 'negative': 20, 'ratio': 0.5}
    '''
    pos_neg_ratio = {'positive': 0, 'negative': 0, 'ratio': 0.0}
    pos_neg_ratio['positive'] = utility_functions.lookup(freqs,word,1)
    pos_neg_ratio['negative'] =utility_functions.lookup(freqs,word,0)
    pos_neg_ratio['ratio'] = (pos_neg_ratio['positive'] + 1)/(pos_neg_ratio['negative'] + 1)
    return pos_neg_ratio

def print_table(word_freqs):
    table = PrettyTable()
    table.field_names = ["Word", "Positive", "Negative", "Ratio"]

    for word, freq in word_freqs.items():
        table.add_row([word, freq['positive'], freq['negative'], freq['ratio']])

    print(table)

def print_words_by_threshold(freqs, label, threshold):
    '''
    Input:
        freqs: dictionary of words
        label: 1 for positive, 0 for negative
        threshold: ratio that will be used as the cutoff for including a word in the returned dictionary
    Output:
        word_set: dictionary containing the word and information on its positive count, negative count, and ratio of positive to negative counts.
        example of a key value pair:
        {'happi':
            {'positive': 10, 'negative': 20, 'ratio': 0.5}
        }
    '''
    word_list = {}

    for key in freqs.keys():
        word, _ = key
        pos_neg_ratio = get_ratio(freqs, word)
        if label == 1 and pos_neg_ratio['ratio'] >= threshold:
            word_list[word] = pos_neg_ratio
        elif label == 0 and pos_neg_ratio['ratio'] <= threshold:
            word_list[word] = pos_neg_ratio
        else:
            pass
            # Because when label in (0,1), do not include this word in the list (do nothing)
            
    print_table(word_list)