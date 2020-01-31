import numpy as np
from scipy.sparse import issparse, isspmatrix
from sklearn import pipeline, feature_extraction, svm, metrics
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC


from traditional_methods import get_data
import time
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
# from sklearn.metrics import F

class Traditional_method:
    def __init__(self, model, model_name="unknown model"):
        # super()
        super().__init__()
        '''
        Build tradition method class
        :param file: csv file
        :param model_name: string
        :param model: sckit learn model
        '''
        self.model = pipeline.Pipeline([

            ('counts', feature_extraction.text.CountVectorizer(
                min_df=5, stop_words="english",analyzer="word", ngram_range=(1,2)
            )
             ),
            ('tfidf', feature_extraction.text.TfidfTransformer()),
            (model_name, model),
        ])
        # self.data = data
        self.model_name = model_name

    def k_fold(self, data_type):
        print(data_type + " data")

        kfold = KFold(5, True, 1)
        # enumerate splits
        X = np.array(data["tweet"])
        y = np.array(data["class"])
        for train_index, test_index in kfold.split(X):
            X_train = X[train_index]
            y_train = y[train_index]

            X_test = X[test_index]
            y_test = y[test_index]
            self.model.fit(X_train, y_train)
            # fit_time = time.time()

            y_pred = self.model.predict(X_test)

            print('Accuracy of ' + self.model_name + '= {}'.format(
                np.mean(y_pred == y_test)))
            print(metrics.classification_report(
                y_test, y_pred, target_names=["hate_speech", "offensive_language", "neither"]))
    # def upsampling(self, data):
    def classification_report(self, X, y , data_type="unbalanced"):
        # data= self.data
        # X, y = data["tweet"], data["class"]
        clf = self.model
        scores = cross_val_score(clf, X, y, cv=5)
        # print("cross validation score", scores)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        # sm = SMOTE()
        # X_train, X_test = sm.fit_sample(X_train, y_train.ravel())
        X_train, y_train = get_data.up_sampling(X_train, y_train)
        start_time = time.time()
        self.model.fit(X_train, y_train)
        fit_time = time.time()

        y_pred = self.model.predict(X_test)

        print(self.model_name + " with " + data_type)
        print(metrics.classification_report(
                y_test, y_pred, target_names=["hate_speech", "offensive_language", "neither"]))

        print("time comsume to fit model", fit_time - start_time)
        print()
        print('Accuracy of '+ self.model_name + 'on test sets= {}'.format(
            np.mean(y_pred == y_test)))
        print()
        X_pred = self.model.predict(X_train)
        print("Accuracy on training set", np.mean(y_train == X_pred))
        print(0)
        print("cross validation score", scores)
        print()

    def classify(self, X, y):
        # self.split(X,y)
        cv = StratifiedKFold(n_splits=5)
        for train_idx, test_idx, in cv.split(X, y):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            # print(np.array(X_train))
            # X_train, y_train = RandomOverSampler().fit_sample(X_train.as_metrix(), y_train.ravel())
            # X_resampled, y_resampled = ros.fit_resample(X, y)

            X_train, y_train = get_data.up_sampling(X_train, y_train)
            self.model.fit(X_train, y_train)
            # fit_time = time.time()

            y_pred = self.model.predict(X_test)
            print(metrics.classification_report(
                y_test, y_pred, target_names=["hate_speech", "offensive_language", "neither"]))


if __name__ == '__main__':
    # data = clean_text(df)
    # data1 = upsampling(clean_text((df)))
    # data2 = downsampling(clean_text(df))
    X, y = get_data.read_file("labeled_data.csv")
    # x_T = []
    # print(list(X))
    # for i in X:
    #     print(i)
    # X, y = SMOTE().fit_sample(X, y)

    # svm = Traditional_method( svm.LinearSVC(), "LinearSVC")
    # svm.classification_report(get_data.upsampling(X, y), "upsampling data")

    linear_svc = Traditional_method(LinearSVC(), "LinearSVC")
    linear_svc.classify(X, y)
