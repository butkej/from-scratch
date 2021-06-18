# Used in Naive Bayes - Spam Classifier

# build a vocabulary of words seen in a given dataset of for example mail

import numpy as np
import pandas as pd
import nltk


vocabulary = {}

data = pd.read_csv('../data/example-emails.csv')
nltk.download('words')

words = set(nltk.corpus.words.words())

def build_vocab(mail, vocabulary=vocabulary):
    idx = len(vocabulary)
    for word in mail:
        if word.lower() not in vocabulary and word.lower() in words:
            vocabulary[word] = idx
            idx += 1


if __name__ == '__main__':
    for i in range(data.shape[0] + 1):
        mail = data.iloc[i,0].split()
        print(f'Current email is {i}/{len(data.shape[0])} and the vocabulary has lenght {len(vocabulary)}')
        build_vocab(mail)

    outfile = open('vocabulary.txt', 'w')
    outfile.write(str(vocabulary))
    outfile.close()
