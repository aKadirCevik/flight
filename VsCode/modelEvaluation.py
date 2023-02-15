from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


class ModelEvaluation:
    # Initializer / Instance Attributes
    def __init__(self, X, y, models):
        self.X = X
        self.y = y
        self.models = models

    # evaluate a list of models on a dataset
    def print_result_of_evaluation(self):
        result = list()
        model_name_count = 0
        yhat_list = self.model_prediction()
        for yhat in yhat_list:
            accuracy = accuracy_score(self.y, yhat)
            precision = precision_score(self.y, yhat)
            recall = recall_score(self.y, yhat)
            f1 = f1_score(self.y, yhat)
            kappa = cohen_kappa_score(self.y, yhat)
            auc = roc_auc_score(self.y, yhat)

            result.append('%s: ' % self.model[model_name_count].__class__.__name__ + '\n' +
                          'Accuracy: %f' % accuracy + '\n' +
                          'Precision: %f' % precision + '\n' +
                          'Recall: %f' % recall + '\n' +
                          'F1 score: %f' % f1 + '\n' +
                          'Cohens kappa: %f' % kappa + '\n' +
                          'ROC AUC: %f' % auc + '\n')
            model_name_count += 1

        print(result)

    def get_model_predictions(self) -> list:
        yhat_list = list()
        for model in self.models:
            yhat_list.append(model.predict(self.X))

        return yhat_list

    def print_confusion_matrix(self):
        model_name_count = 0
        yhat_list = self.get_model_predictions()
        for yhat in yhat_list:
            matrix = confusion_matrix(self.y, yhat)
            print('%s: ' % self.model[model_name_count].__class__.__name__)
            print(matrix)
            model_name_count += 1

    def get_confusion_matrix(self):
        yhat_list = self.get_model_predictions()
        for yhat in yhat_list:
            matrix = confusion_matrix(self.y, yhat)

        return matrix
