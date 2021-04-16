import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import  LogisticRegression
class LogisticRegression1():

    """sigmoid function  transforms that input value to a range [0,1]
    In our case, anything below 0.5 will be mapped to 0, and anything above or equal to 0.5 will be mapped to 1.
   """
    def sigmoid(self, score):
        return (1/(1 + np.exp(-score)))
    def predict_probability(self, feature, weight):
        score = np.dot(feature, weight)
        return self.sigmoid(score)
    def feature_derivative(self, errors, feature, weight, l2_penality, feature_is_contant):
        derivative = np.dot(np.transpose(errors), feature)
        if not feature_is_contant:
            derivative -= 2 * l2_penality * weight
        return derivative
    def compute_log_likelihood(self, features, labels, weights, l2_penalty):
        indicator = (labels==+1)
        scores = np.dot(features, weights)
        logLikelihood = np.sum((np.transpose(np.array([indicator]))-1) * scores-np.log(1.+np.exp(-scores))) - (l2_penalty * np.sum(weights[1:]**2))
        return logLikelihood
    def fit(self, features, labels, lr, epochs, l2_penality):
        bias = np.ones((features.shape[0], 1))
        features = np.hstack((bias, features))
        weights = np.zeros((features.shape[1],1))
        logs = []
        for epoch in range(epochs):
            yhat = self.predict_probability(features, weights)
            indicators = (labels == +1)
            errors = np.transpose(np.array([indicators])) - yhat
            for j in range(len(weights)):
                isIntercept = (j==0)
                dervative = self.feature_derivative(errors, features[:,j],weights[j], l2_penality,isIntercept)
                weights[j]+=lr * dervative
            l1 = self.compute_log_likelihood(features, labels, weights, l2_penality)
            logs.append(l1)
        return weights
    def score(self, learned_weights, test_features):
        bias = np.ones((test_features.shape[0], 1))
        features = np.hstack((bias, test_features))
        return (self.predict_probability(features, learned_weights).flatten()>0.5)

if __name__ == '__main__':

    data = load_breast_cancer()
    #print(data.keys())
    #print(data['target_names'])
    df = pd.DataFrame(data.data)
    #print(df.describe() , '\n')
    #print(df.columns)
    X_train, X_test, y_train,y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=9)
    learing_rate = 1e-7
    epochs = 30000
    l2_penality = 0.001
    l_reg = LogisticRegression1()
    l_weights = l_reg.fit(X_train, y_train, learing_rate, epochs, l2_penality)
    test_predictions =  l_reg.score(l_weights, X_test)
    train_predictions = l_reg.score(l_weights, X_train)
    print("Accuracy of model on test data: {}".format(accuracy_score( np.expand_dims(y_test, axis =1), test_predictions)))
    print("Accuracy of model on Train data: {}".format(accuracy_score( np.expand_dims(y_train, axis =1), train_predictions)))
    model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter= 30000)
    model.fit(X_train, y_train)
    sk_test_predictions = model.predict(X_test)
    sk_train_predictions = model.predict(X_train)
    print("Accuracy of scikit-learn's LR classifier on training data: {}".format(
        accuracy_score(y_train, sk_train_predictions)))
    print("Accuracy of scikit-learn's LR classifier on testing data: {}".format(
        accuracy_score(y_test, sk_test_predictions)))




