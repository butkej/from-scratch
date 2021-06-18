# Simple implementation of a Gaussian naive bayes classifier (eg. for spam mail detection)

import numpy as np

class NaiveBayes():
    def __init__(self, X, y):
        self.num_samples, self.num_features = X.shape # for X with shape of (rows, columns) ie (samples,features)
        self.num_classes = len(np.unique(y))
        self.epsilon = 1e-8

    def train(self, X, y):
        self.classes_mean = {}
        self.classes_var = {}
        self.classes_prior = {}

        for c in range(self.num_classes):
            X_c = X[y==c] # pick out samples from a specific class
            self.classes_mean[str(c)] = np.mean(X_c, axis=0)
            self.classes_var[str(c)] = np.var(X_c, axis=0)
            self.classes_prior[str(c)] = X_c.shape[0]/self.num_samples


    def predict(self, X):
        # Return a probabilty for each sample, indicating to which class it belongs
        probabilities = np.zeros((self.num_samples, self.num_classes))

        for c in range(self.num_classes):
            prior = self.classes_prior[str(c)]
            probabilites_c = self.density_func(X, self.classes_mean[str(c)], self.classes_var[str(c)])
            probabilities[:, c] = probabilites_c + np.log(prior)

        return np.argmax(probabilities, 1)

    def density_func(self, X, mu, sigma):
        # calculate probability from a gaussian density function
        constant = -self.num_features/2 * np.log(2*np.pi) - 0.5*np.sum(np.log(sigma+self.epsilon))
        probabilities = 0.5*np.sum(np.power(X-mu, 2) / (sigma+self.epsilon), 1)
        return constant - probabilities


###

if __name__ == "__main__":

    classify_spam_in_email = True # flag to determina the data that is loaded: True for spam mail classification

    if not classify_spam_in_email:
        X = np.loadtxt('data/example-data.txt', delimiter=',')
        y = np.loadtxt('data/example-targets.txt') - 1

    elif classify_spam_in_email:
        X = np.load('data/X-spam.npy')
        y = np.load('data/y-spam.npy')

    print(X.shape)
    print(y.shape)

    NBClassifier = NaiveBayes(X, y)

    NBClassifier.train(X,y)
    y_pred = NBClassifier.predict(X)

    print('Accuracy ',sum(y_pred == y)/X.shape[0])
