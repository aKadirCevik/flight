from numpy import hstack
from numpy import vstack
from numpy import asarray
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd


class Superlearner:
    # Initializer / Instance Attributes
    def __init__(self, f_path):
        # self.f_path = f_path
        self.f_path = "C:\\Users\\nursa\\Documents\\Nursah\\TİK 3\\Flight\\VsCode\\data\\resultData\\result_dataframe.csv"
        self.models = list()
        self.meta_X, self.meta_y = list(), list()

    # create a list of base-models
    def get_models(self):
        self.models.append(LogisticRegression(solver='liblinear'))
        self.models.append(DecisionTreeClassifier())
        # self.models.append(SVC(gamma='scale', probability=True))
        self.models.append(GaussianNB())
        self.models.append(KNeighborsClassifier())
        # self.models.append(AdaBoostClassifier())
        # self.models.append(BaggingClassifier(n_estimators=10))
        self.models.append(RandomForestClassifier(n_estimators=10))
        self.models.append(ExtraTreesClassifier(n_estimators=10))

    @property
    def create_dataset(self):
        dataset = pd.read_csv(self.f_path, sep=',', header=0)

        X = dataset.iloc[:, 1:-1].values
        y = dataset.iloc[:, -1]
        # y = y.astype('string')

        return X, y

    # collect out of fold predictions form k-fold cross validation
    def get_out_of_fold_predictions(self, X, y):
        # enumerate splits
        kfold = KFold(n_splits=3, shuffle=True)
        for train_ix, test_ix in kfold.split(X):
            fold_yhats = list()
            # get data
            train_X, test_X = X[train_ix], X[test_ix]
            train_y, test_y = y[train_ix], y[test_ix]
            self.meta_y.extend(test_y)
           
            # fit and make predictions with each sub-model
            for model in self.models:
                model.fit(train_X, train_y)
                yhat = model.predict_proba(test_X)
                # store columns
                fold_yhats.append(yhat)
            # store fold yhats as columns
            self.meta_X.append(hstack(fold_yhats))
        return vstack(self.meta_X), asarray(self.meta_y)

    # fit all base models on the training dataset
    def fit_base_models(self, X, y):
        for model in self.models:
            model.fit(X, y)

    # fit a meta model
    @staticmethod
    def fit_meta_model(X, y):
        model = LogisticRegression(solver='liblinear')
        model.fit(X, y)
        return model

    # evaluate a list of models on a dataset
    def evaluate_models(self, X, y):
        for model in self.models:
            yhat = model.predict(X)
            acc = accuracy_score(y, yhat)
            print('%s: %.3f' % (model.__class__.__name__, acc*100))

    # make predictions with stacked model
    def super_learner_predictions(self, X, meta_model):
        self.meta_X = list()
        for model in self.models:
            yhat = model.predict_proba(X)
            self.meta_X.append(yhat)
        self.meta_X = hstack(self.meta_X)
        # predict
        return meta_model.predict(self.meta_X)

    def run(self):
        # create the inputs and outputs
        X, y = self.create_dataset

        # split
        X, X_val, y, y_val = train_test_split(X, y, test_size=0.30)
        y = y.to_numpy()
        print('Train', X.shape, y.shape, 'Test', X_val.shape, y_val.shape)

        # get models
        self.get_models()

        # get out of fold predictions
        self.meta_X, self.meta_y = self.get_out_of_fold_predictions(X, y)
        print('Meta ', self.meta_X.shape, self.meta_y.shape)

        # fit base models
        self.fit_base_models(X, y)

        # fit the meta model
        meta_model = self.fit_meta_model(self.meta_X, self.meta_y)

        # evaluate base models
        self.evaluate_models(X_val, y_val)

        # evaluate meta model
        yhat = self.super_learner_predictions(X_val, meta_model)
        print('Super Learner: %.3f' % (accuracy_score(y_val, yhat) * 100))


