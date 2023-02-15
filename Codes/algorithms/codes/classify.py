from sklearn.utils import shuffle
from sklearn.datasets import make_blobs
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import os
import sys


class Classify:

    # Initializer / Instance Attributes
    def __init__(self, k):
        self.models = list()
        # define split of data
        self.k = k


    # create a list of base-models
    def get_models(self):
        self.models.append(LogisticRegression(solver='liblinear'))
        self.models.append(DecisionTreeClassifier())
        self.models.append(GaussianNB())
        self.models.append(KNeighborsClassifier())
        self.models.append(ExtraTreesClassifier(n_estimators=10))
        self.models.append(RandomForestClassifier(n_estimators=500, verbose=1, n_jobs=-1))
        # self.models.append(SVC(gamma='scale', probability=True))
        # self.models.append(AdaBoostClassifier())
        # self.models.append(BaggingClassifier(n_estimators=10))

    def create_dataset(self):
        sys.path.append(os.getcwd()+'\\resultDF.csv')  
        path = sys.path[-1]
        dataset = pd.read_csv(path, sep=',', header=0)
        
        # X, y = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=0)
   
        X = dataset.iloc[:, 0:-2].values
        y = dataset.iloc[:, -1]
        y = y.astype('string')

        return X, y

    # collect out of fold predictions form k-fold cross validation
    def get_out_of_fold_predictions(self, X, y):
        kfold = KFold(n_splits = self.k, shuffle=True)
        # enumerate splits
        for train_ix, test_ix in kfold.split(X):
            fold_yhats = list()
            # get data
            train_X, test_X = X[train_ix], X[test_ix]
            train_y, test_y = y[train_ix], y[test_ix]
           
            # fit and make predictions with each sub-model
            for model in self.models:
                model.fit(train_X, train_y)
                yhat = model.predict_proba(test_X)
                # store columns
                fold_yhats.append(yhat)



    # evaluate a list of models on a dataset
    def evaluate_models(self, X, y):
        result = list()
        for model in self.models:
            yhat = model.predict(X)
            acc = accuracy_score(y, yhat)
            result.append('%s: %.3f' % (model.__class__.__name__, acc*100))
        return result

    def main(self):
        # create the inputs and outputs
        X, y = self.create_dataset() 

        # split
        X, X_val, y, y_val = train_test_split(X, y, test_size=0.30)
        y = y.to_numpy()
        print('Train', X.shape, y.shape, 'Test', X_val.shape, y_val.shape)

        # get models
        self.get_models()

        # get out of fold predictions
        self.get_out_of_fold_predictions(X, y)
    

        # evaluate base models
        result = self.evaluate_models(X_val, y_val)
        print("MODEL RESULTS: ")
        for item in result:
            print(item)


mn = Classify(3)
mn.main()