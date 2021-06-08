import numpy as np

class LogisticRegression():
    def __init__(self, lr=0.1, n_iter=10000):
        self.lr = lr
        self.n_iter = n_iter
        

    def fit(self, X, y):

        self.examples, self.features = X.shape

        # initialize weights and bias 
        self.weights = np.zeros((self.features, 1))
        self.bias = 0
        
        for i in range(self.n_iter + 1):
            # calculate hypothesis
            y_predict = self.logistic(np.dot(X, self.weights) + self.bias)
            # calculate loss
            loss = -1/self.examples * np.sum(y * np.log(y_predict) + (1-y) * np.log(1-y_predict))
            # update parameters
            dw = 1/self.examples * np.dot(X.T, (y_predict - y))
            db = 1/self.examples * np.sum(y_predict - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if i % 1000 == 0:
                print(f'Loss after iteration step {i}: {loss}')
            

    def predict(self, X):

        y_predict = self.logistic(np.dot(X, self.weights) + self.bias)
        y_predict_label = y_predict > 0.5
        return y_predict_label

    def logistic(self, z):
        ''' returns the logistic funcion of its input or
        rather the special case of a sigmoid function
        '''
        return 1 / (1 + np.exp(-z))



if __name__ == '__main__':

    from sklearn.datasets import make_blobs
    from sklearn.model_selection import train_test_split

    # build dataset
    X, y = make_blobs(n_samples=1000, centers=2)
    y = y[:, np.newaxis] # transform (n_examples,) to (n_examples,1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f'Accuracy is: {np.sum(preds == y_test)/X_test.shape[0]}')
