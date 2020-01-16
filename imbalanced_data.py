import re

import numpy as np
import pandas as pd
from sklearn import pipeline, feature_extraction, svm, metrics
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("labeled_data.csv")

def  clean_text(df, text_field="tweet"):
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
    return df

def downsampling(data):
    hate_speech = data[data["class"] == 0]
    offensive_language = data[data["class"] == 1]
    neither = data[data["class"] == 2]
    neither = resample(neither,
                        replace=True,
                        n_samples=len(hate_speech),
                        random_state=123)
    offensive_language = resample(offensive_language,
                                        replace=True,
                                        n_samples=len(hate_speech),
                                        random_state=123)
    downsampled = pd.concat([hate_speech, offensive_language, neither])
    downsampled["class"].value_counts()
    return downsampled

def upsampling(data):
    hate_speech = data[data["class"] == 0]
    offensive_language = data[data["class"] == 1]
    neither = data[data["class"] == 2]
    hate_speech = resample(hate_speech,
                                        replace=True,
                                        n_samples=len(neither),
                                        random_state=123)
    offensive_language = resample(offensive_language,
                                        replace=True,
                                        n_samples=len(neither),
                                        random_state=123)
    upsampled = pd.concat([hate_speech, offensive_language, neither])
    upsampled["class"].value_counts()
    return upsampled
if __name__ == '__main__':
    data = clean_text(df)
    data1 = upsampling(clean_text((df)))
    data2 = downsampling(clean_text(df))

    model = pipeline.Pipeline([
        ('counts', feature_extraction.text.CountVectorizer()),
        ('tfidf', feature_extraction.text.TfidfTransformer()),
        ('svm', svm.LinearSVC()),
    ])

    X_train, X_test, y_train, y_test = train_test_split(data['tweet'], data['class'],random_state = 0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("unbalanced")
    print('Accuracy of SVM= {}'.format(
        np.mean(y_pred == y_test)))

    print(metrics.classification_report(
        y_test, y_pred, target_names=["hate_speech", "offensive_language", "neither"]))

    print("upsampling")

    X_train, X_test, y_train, y_test = train_test_split(data1['tweet'], data1['class'],random_state = 0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print('Accuracy of SVM= {}'.format(
        np.mean(y_pred == y_test)))

    print(metrics.classification_report(
        y_test, y_pred, target_names=["hate_speech", "offensive_language", "neither"]))


    print("downsampling")
    X_train, X_test, y_train, y_test = train_test_split(data2['tweet'], data2['class'],random_state = 0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print('Accuracy of SVM= {}'.format(
        np.mean(y_pred == y_test)))

    print(metrics.classification_report(
        y_test, y_pred, target_names=["hate_speech", "offensive_language", "neither"]))
