from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from modelEvaluation import ModelEvaluation


class Classify:
    # Initializer / Instance Attributes
    def __init__(self, f_path):
        self.f_path = f_path
        self.models = list()
        self.train_mean = 0
        self.train_std = 0
        # self.model_names = ["Logistic Regression", "Decision Tree Classifier",
        # "GaussianNB", "KNeighborsClassifier", "ExtraTreesClassifier",
        # "RandomForestClassifier"]

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
        dataset = pd.read_csv(self.f_path, sep=',', header=0)
        dataset = shuffle(dataset)
        dataset = dataset.reset_index()
        del dataset['index']
        # print("Length: ", len(dataset))

        return dataset

    @staticmethod
    def split_data_x_and_y(dataset):
        X = dataset.iloc[:, 1:-1].values
        y = dataset.iloc[:, -1]

        return X, y

    def pre_processing_train_values(self, features, labels):
        # get per-feature statistics (mean, standard deviation) from the training set to normalize by
        self.train_mean = np.nanmean(features, axis=0)
        self.train_std = np.nanstd(features, axis=0)
        features = (features - self.train_mean) / self.train_std
        df = pd.DataFrame(features)

        # organize train labels and merge with train features
        labels = labels.reset_index()
        del labels['index']
        df["label"] = labels

        # Drop Nan values
        df = df.dropna()
        df = df.reset_index()
        del df['index']

        features = df.iloc[:, 0:-1].values
        labels = df.iloc[:, -1]

        return features, labels

    def pre_processing_test_values(self, features, labels):
        features = (features - self.train_mean) / self.train_std
        df = pd.DataFrame(features)

        # organize train labels and merge with train features
        labels = labels.reset_index()
        del labels['index']
        df["label"] = labels

        # Drop Nan values
        df = df.dropna()
        df = df.reset_index()
        del df['index']

        features = df.iloc[:, 0:-1].values
        labels = df.iloc[:, -1]

        return features, labels

    # collect out of fold predictions form k-fold cross validation
    def get_out_of_fold_predictions(self, train_X, train_y):
        # fit and make predictions with each sub-model
        for model in self.models:
            model.fit(train_X, train_y)

    # evaluate a list of models on a dataset
    def evaluate_models_old(self, X, y):
        result = list()
        for model in self.models:
            yhat = model.predict(X)
            acc = accuracy_score(y, yhat)
            result.append('%s: %.3f' % (model.__class__.__name__, acc * 100))

        return result

    # evaluate a list of models on a dataset
    def evaluate_models(self, X, y):
        result = list()
        for model in self.models:
            yhat = model.predict(X)

            # accuracy: (tp + tn) / (p + n)
            accuracy = accuracy_score(y, yhat)
            # print('Accuracy: %f' % accuracy)
            # precision tp / (tp + fp)
            precision = precision_score(y, yhat)
            # print('Precision: %f' % precision)
            # recall: tp / (tp + fn)
            recall = recall_score(y, yhat)
            # print('Recall: %f' % recall)
            # f1: 2 tp / (2 tp + fp + fn)
            f1 = f1_score(y, yhat)
            # print('F1 score: %f' % f1)
            # kappa
            kappa = cohen_kappa_score(y, yhat)
            # print('Cohens kappa: %f' % kappa)
            # ROC AUC
            auc = roc_auc_score(y, yhat)
            # print('ROC AUC: %f' % auc)

            # confusion matrix
            matrix = confusion_matrix(y, yhat)
            print('%s: ' % model.__class__.__name__)
            print(matrix)
            # plot_confusion_matrix(model, X, y)
            # plt.show()

            result.append('%s: ' % model.__class__.__name__ + '\n' +
                          'Accuracy: %f' % accuracy + '\n' +
                          'Precision: %f' % precision + '\n' +
                          'Recall: %f' % recall + '\n' +
                          'F1 score: %f' % f1 + '\n' +
                          'Cohens kappa: %f' % kappa + '\n' +
                          'ROC AUC: %f' % auc + '\n')
        return result

    def make_prediction(self):
        # create the inputs and outputs
        test_features, test_labels, dataset = self.create_dataset()
        print('Test', test_features.shape, test_labels.shape)

        # ------------------------------------------------------------------------------------------------
        test_features, test_labels = self.pre_processing_test_values(test_features, test_labels)

        # evaluate base models
        result = self.evaluate_models(test_features, test_labels)
        print("MODEL RESULTS: ")
        for item in result:
            print(item)

    def run(self):
        # create the inputs and outputs
        dataset = self.create_dataset()
        X, y = self.split_data_x_and_y(dataset)

        # split
        train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.30)
        print('Train', train_features.shape, train_labels.shape, 'Test', test_features.shape, test_labels.shape)

        # ------------------------------------------------------------------------------------------------
        train_features, train_labels = self.pre_processing_train_values(train_features, train_labels)
        test_features, test_labels = self.pre_processing_test_values(test_features, test_labels)

        # get models
        self.get_models()

        # get out of fold predictions
        self.get_out_of_fold_predictions(train_features, train_labels)

        # evaluate base models
        result = self.evaluate_models(test_features, test_labels)
        print("MODEL RESULTS: ")
        for item in result:
            print(item)
