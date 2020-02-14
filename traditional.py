import numpy as np
from scipy.sparse import issparse, isspmatrix
from sklearn import pipeline, feature_extraction, svm, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from traditional_methods import get_data
import time
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler


class sm(SMOTE):
    def fit(self, X, y):
        return self.fit_resample(X, y)
    def transform(self, X, y):
        return X, y

class Traditional_method():
    def __init__(self, model, model_name="unknown model"):
        '''
        Build tradition method class
        :param file: csv file
        :param model_name: string
        :param model: sckit learn model
        '''
        self.model = pipeline.Pipeline([
            ('counts', feature_extraction.text.CountVectorizer(
                min_df=5, stop_words="english",analyzer="word", ngram_range=(1,2)
            )),
            ('tfidf', feature_extraction.text.TfidfTransformer()),
            (model_name, model) ])

        self.model_name = model_name


    def classify(self, X, y):
        # self.split(X,y)
        cv = StratifiedKFold(n_splits=5)
        results = [0,0,0, 0,0,0,0,0,0]
        for train_idx, test_idx, in cv.split(X, y):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_test)

            results = [x + y for x, y in zip(results , list( metrics.precision_recall_fscore_support(y_test, y_pred, average="macro")[:-1] + metrics.precision_recall_fscore_support(y_test, y_pred, average="micro")[:-1] + metrics.precision_recall_fscore_support(y_test, y_pred, average="weighted")[:-1]))]
            print(results)
        print(self.model_name, [round(x / 5, 2) for x in results ])

if __name__ == '__main__':

    data  = get_data.read_file("labeled_data.csv")

    linear_svc = Traditional_method(LinearSVC(), "LinearSVC")
    X, y = data
    linear_svc.classify(X, y)

    logistic_regression_method = Traditional_method(LogisticRegression(max_iter=1000),"logistic regression")
    logistic_regression_method.classify(data[0], data[1])

    gaussion_nb_method = Traditional_method(MultinomialNB(), "MultinomialNB")
    gaussion_nb_method.classify(data[0], data[1])

    gaussion_nb_method =Traditional_method(MultinomialNB(), "MultinomialNB")
    gaussion_nb_method.classify(data[0], data[1])

    decision_tree_method = Traditional_method(DecisionTreeClassifier(), "DecisionTreeClassifier")
    decision_tree_method.classify(data[0], data[1])

    random_forest_method = Traditional_method(RandomForestClassifier(), "RandomForestClassifier")
    random_forest_method.classify(data[0], data[1])
