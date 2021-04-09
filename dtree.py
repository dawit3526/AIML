
import numpy as np
from collections import Counter


class DecisionNode:

    def __init__(self, col=None, split=None, lchild=None, rchild=None,*, value=None):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild
        self.value = value
    def leaf_node(self):
        return self.value is not None

class DecisionTree:

    def __init__(self, min_samples_leaf=1, loss=None):
        self.min_samples_leaf = min_samples_leaf
#         self.loss = loss # loss function; either np.std or gini

    def fit(self, X, y):
        """
        Create a decision tree fit to (X,y) and save as self.root, the root of
        our decision tree, for either a classifier or regressor.  Leaf nodes for classifiers
        predict the most common class (the mode) and regressors predict the average y
        for samples in that leaf.  
              
        This function is a wrapper around fit_() that just stores the tree in self.root.
        """
        self.root = self.fit_(X, y, self.min_samples_leaf)
        
    
    def fit_(self, X, y, min_samples_leaf):
        """
        Recursively create and return a decision tree fit to (X,y) for
        either a classifier or regressor.  This function should call self.create_leaf(X,y)
        to create the appropriate leaf node, which will invoke either
        RegressionTree621.create_leaf() or ClassifierTree621. create_leaf() depending
        on the type of self.
        
        This function is not part of the class "interface" and is for internal use, but it
        embodies the decision tree fitting algorithm.

        (Make sure to call fit_() not fit() recursively.)
        """
        samples, features = X.shape 

        if samples < self.min_samples_leaf:
            leaf = self.create_leaf(y)
            return DecisionNode(value=leaf)
        
        
        best_col, best_split = self.best_split(X, y)
        if best_col == -1:
            leaf = self.create_leaf(y)
            return DecisionNode(value=leaf)

#                   
        lchild = self.fit_(X[X[:, best_col] <= best_split], y[X[:, best_col] <= best_split], self.min_samples_leaf)
        rchild = self.fit_(X[X[:, best_col] > best_split], y[X[:, best_col] > best_split], self.min_samples_leaf)
        return DecisionNode(best_col, best_split, lchild, rchild)

    def best_split(self, X, y):
        best_gain = -1
        best_col, best_split = -1,-1
        
        for col in range(X.shape[1]):
            X_column = X[:, col]
            candidates = np.random.choice(X_column, size=11, replace=True)
            # print(candidates)
            candidates = np.unique(candidates)
            thresholds = X_column
            
            for split in candidates:
                yl = y[X[:, col] <= split]
                yr = y[X[:, col] > split]

                loss = self.loss(y)

                if len(yl) < self.min_samples_leaf  or  len(yr) < self.min_samples_leaf:
                    continue

                child_loss = (len(yl) / len(y)) * self.loss(yl) + (len(yr) / len(y)) * self.loss(yr)
                l = loss - child_loss

                
                if l == 0:
                    return best_col, best_split
                if l > best_gain:
                    best_gain = l
                    best_col = col
                    best_split = split
        
        return best_col, best_split

    def predict(self, X):
        
        return np.array([self.traverse_tree(x, self.root) for x in X])

    def traverse_tree(self, x, node):
        if node.leaf_node():
            return node.value

        if x[node.col] <= node.split:
            return self.traverse_tree(x, node.lchild)
        return self.traverse_tree(x, node.rchild)

    


class RegressionTree621(DecisionTree):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf)

    def score(self, X_test, y_test):
        "Return the R^2 of y_test vs predictions for each record in X_test"
        y_pred = self.predict(X_test)
        y_mean = np.mean(y_test)
        residuals = y_pred - y_test
        SSE = np.sum(np.power(residuals, 2))
        SST = np.sum(np.power(y_test-y_mean, 2))
        r_squared = 1 - SSE/SST
        
        return r_squared
    
    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """    
        return np.mean(y)
    
    def loss(self,y):
        
        return np.std(y)
#     
class ClassifierTree621(DecisionTree):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf)

    def score(self, X_test, y_test):
        "Return the accuracy_score() of y_test vs predictions for each record in X_test"
        y_pred = self.predict(X_test)
        
        accuracy = np.sum(y_test == y_pred) / len(y_test)
        return accuracy
#     
    def loss(self,y):
        _, counts = np.unique(y, return_counts=True)
    
        n = len(y)
        return 1 - np.sum( (counts / n)**2 )
    
    def create_leaf(self, y):
        return np.bincount(y).argmax()
        




