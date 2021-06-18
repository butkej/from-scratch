''' helper script to go through each mail and map into an X, y dataset that will be the
input of an naive bayes spam classifier

Each mail will result in a list of unique words and their corresponding frequency
[apple, banana, ..., vuvuzela]
[5, 10, ..., 2]
'''

import numpy as np
import pandas as pd
import ast


data = pd.read_csv('../../data/example-emails.csv')
vocab = open('vocabulary.txt', 'r').read()
vocab = ast.literal_eval(vocab)

X = np.zeros((data.shape[0], len(vocab)))
y = np.zeros((data.shape[0]))

for i in range(data.shape[0] +1):
    mail = data.iloc[i,0].split()
    
    for word in mail:
        if word.lower() in vocab:
            X[i, vocab[word]] += 1
            y[i] = data.iloc[i,1]


np.save('../data/X-spam.npy', X)
np.save('../data/y-spam.npy', y)

